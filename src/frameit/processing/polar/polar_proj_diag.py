# Copyright 2026 Clément Soufflet, Météo-France
# Licensed under the Apache License, Version 2.0
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from frameit.processing.polar.polar_grid import PolarLonLatGrid
from frameit.processing.requests import build_group_ds, normalize_requests

try:
    import xesmf as xe
except Exception as exc:  # pragma: no cover
    xe = None
    _XESMF_IMPORT_ERROR = exc
else:
    _XESMF_IMPORT_ERROR = None


def _log(logger: logging.Logger | None, level: str, msg: str) -> None:
    if logger is None:
        print(msg)
        return
    getattr(logger, level)(msg)


def _as_lower_str(x: Any) -> str:
    return str(x).strip().lower()


def _get_required(conf: Any, name: str) -> Any:
    v = getattr(conf, name, None)
    if v is None:
        raise ValueError(f"Missing required conf attribute: {name!r}")
    return v


def _get_polar_requests(conf: Any) -> dict[str, dict[str, Any]]:
    """
    Build normalized polar-variable requests from the configuration.

    Parameters
    ----------
    conf : object
        Configuration object.  Reads ``polar_variables``, which may be:

        - ``"all"`` : project all variables from ``conf.requested_variables_user``.
        - A dict shaped like ``requested_variables_user`` : project only those
          variables, with full level-selection normalization applied.

    Returns
    -------
    dict[str, dict[str, Any]]
        Normalized request dict as returned by
        :func:`~frameit.processing.requests.normalize_requests`.
        Keys are vertical group names; values contain ``"variables"``,
        ``"all_levels"``, ``"level_indices"``, and ``"levels"`` entries.
    """
    raw = getattr(conf, "polar_variables", None)
    if isinstance(raw, str) and _as_lower_str(raw) == "all":
        raw = getattr(conf, "requested_variables_user", None)
    return normalize_requests(raw)


def _polar_lonlat_from_grid(ds_polar: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Extract longitude and latitude arrays from a polar grid dataset.

    Parameters
    ----------
    ds_polar : xr.Dataset
        Polar grid dataset produced by
        :meth:`~frameit.processing.polar.polar_grid.PolarLonLatGrid.build`.
        Accepts both ``"lon"``/``"lat"`` and ``"longitude"``/``"latitude"``
        naming conventions.

    Returns
    -------
    lon : xr.DataArray
        Longitude of the polar grid points.
    lat : xr.DataArray
        Latitude of the polar grid points.

    Raises
    ------
    ValueError
        If neither ``"lon"`` nor ``"longitude"`` (or ``"lat"``/``"latitude"``)
        are present in ``ds_polar``.
    """
    if "lon" in ds_polar:
        lon = ds_polar["lon"]
    elif "longitude" in ds_polar:
        lon = ds_polar["longitude"]
    else:
        raise ValueError(
            "Polar grid dataset missing longitude field (expected 'lon' or 'longitude')."
        )

    if "lat" in ds_polar:
        lat = ds_polar["lat"]
    elif "latitude" in ds_polar:
        lat = ds_polar["latitude"]
    else:
        raise ValueError(
            "Polar grid dataset missing latitude field (expected 'lat' or 'latitude')."
        )

    return lon, lat


def _reshape_locstream_dataset(
    ds_loc: xr.Dataset,
    *,
    r_km: np.ndarray,
    theta_deg: np.ndarray,
    loc_dim: str = "locations",
) -> xr.Dataset:
    """
    Reshape xESMF LocStream output into a polar 2-D ``(r_km, theta_deg)`` grid.

    Parameters
    ----------
    ds_loc : xr.Dataset
        LocStream dataset produced by xESMF, where variables have a flat
        ``loc_dim`` dimension of length ``len(r_km) * len(theta_deg)``.
    r_km : np.ndarray
        Radial distances in kilometres, shape ``(nr,)``.
    theta_deg : np.ndarray
        Azimuths in degrees, shape ``(ntheta,)``.
    loc_dim : str, optional
        Name of the flat locstream dimension in ``ds_loc``. Default ``"locations"``.

    Returns
    -------
    xr.Dataset
        Dataset with the locstream dimension replaced by ``"r_km"`` and
        ``"theta_deg"`` dimensions.  Other dimensions (time, level) are
        preserved.

    Raises
    ------
    ValueError
        If ``loc_dim`` is absent from ``ds_loc`` or its size does not match
        ``len(r_km) * len(theta_deg)``.
    """
    if loc_dim not in ds_loc.dims:
        raise ValueError(f"Expected locstream dimension {loc_dim!r} in xESMF output.")

    nr = int(len(r_km))
    nt = int(len(theta_deg))
    nloc = int(ds_loc.sizes[loc_dim])
    if nloc != nr * nt:
        raise ValueError(f"LocStream size mismatch: {nloc} != {nr}*{nt}")

    out_vars: dict[str, xr.DataArray] = {}

    for v in ds_loc.data_vars:
        da = ds_loc[v]
        if loc_dim not in da.dims:
            out_vars[v] = da
            continue

        if da.dims[-1] != loc_dim:
            other = [d for d in da.dims if d != loc_dim]
            da = da.transpose(*other, loc_dim)

        other_dims = list(da.dims[:-1])
        other_shape = [da.sizes[d] for d in other_dims]
        new_shape = tuple(other_shape + [nr, nt])

        data = da.data.reshape(new_shape)
        new_dims = tuple(other_dims + ["r_km", "theta_deg"])

        coords: dict[str, Any] = {}
        for d in other_dims:
            if d in da.coords:
                coords[d] = da.coords[d]
        coords["r_km"] = ("r_km", r_km)
        coords["theta_deg"] = ("theta_deg", theta_deg)

        out_vars[v] = xr.DataArray(
            data=data,
            dims=new_dims,
            coords=coords,
            attrs=da.attrs,
            name=da.name,
        )

    return xr.Dataset(out_vars)


def _fmt_dim_sizes(ds: xr.Dataset) -> str:
    parts = [f"{k}={int(v)}" for k, v in ds.sizes.items()]
    return ", ".join(parts)


def _describe_coord(da: xr.DataArray, *, max_listed: int = 12) -> str:
    """Short, safe string for a coordinate array."""
    try:
        dtype = str(da.dtype)
        dims = ",".join(map(str, da.dims)) if da.dims else "()"
        shape = "x".join(str(int(s)) for s in da.shape) if da.shape else "()"

        if da.ndim == 0:
            return f"dims={dims}, shape={shape}, dtype={dtype}, value={da.values!r}"

        vals = np.asarray(da.values)
        head = vals.ravel()[:max_listed]

        if np.issubdtype(vals.dtype, np.number):
            vmin = np.nanmin(vals) if vals.size else np.nan
            vmax = np.nanmax(vals) if vals.size else np.nan
            return (
                f"dims={dims}, shape={shape}, dtype={dtype}, "
                f"min={vmin!r}, max={vmax!r}, head={head!r}"
            )

        return f"dims={dims}, shape={shape}, dtype={dtype}, head={head!r}"

    except Exception as exc:  # pragma: no cover
        return f"<coord describe failed: {type(exc).__name__}: {exc}>"


def _horizontal_dims_from_lonlat(
    ds: xr.Dataset, lon_name: str, lat_name: str
) -> tuple[tuple[str, ...], str]:
    """Infer horizontal dims from lon/lat arrays, fallback to common names."""
    for name in (lon_name, lat_name, "lon", "lat", "longitude", "latitude"):
        if name in ds and getattr(ds[name], "ndim", 0) >= 2:
            return tuple(ds[name].dims[-2:]), name

    for cand in (("y_box", "x_box"), ("y", "x"), ("rlat", "rlon")):
        if all(d in ds.dims for d in cand):
            return cand, "fallback"

    return tuple(), "unknown"


def _dataset_scalar_dim_conflicts(ds: xr.Dataset) -> dict[str, str]:
    """
    Return dims that also exist as scalar variables or scalar coords.
    This is the signature pattern of:
      ValueError: dimension 'X' already exists as a scalar variable
    """
    out: dict[str, str] = {}
    for d in ds.dims:
        if d in ds.variables and tuple(ds[d].dims) == ():
            kind = "coord" if d in ds.coords else "data_var"
            out[d] = kind
    return out


def _diagnose_group_datasets(
    *,
    group_key: str,
    ds_src: xr.Dataset,
    ds_sub: xr.Dataset,
    lon_name: str,
    lat_name: str,
    time_dim: str,
    logger: logging.Logger | None,
    max_vars: int = 50,
    max_levels_listed: int = 30,
    compute_level_counts: bool = True,
) -> dict[str, Any]:
    """
    Emit AROME-aware diagnostics for polar projection input datasets.

    Inspects source and subset datasets for scalar-dimension conflicts,
    missing horizontal/vertical coordinates, and all-NaN vertical levels.
    Does not modify any input.

    Parameters
    ----------
    group_key : str
        Vertical group identifier (e.g. ``"level"``, ``"heightAboveGround"``).
    ds_src : xr.Dataset
        Full source dataset for the group (before variable selection).
    ds_sub : xr.Dataset
        Subset dataset containing only requested variables.
    lon_name : str
        Name of the longitude coordinate in ``ds_src``.
    lat_name : str
        Name of the latitude coordinate in ``ds_src``.
    time_dim : str
        Name of the time dimension.
    logger : logging.Logger or None
        Logger to use.  When ``None``, messages are printed to stdout.
    max_vars : int, optional
        Maximum number of variables to inspect for all-NaN levels. Default 50.
    max_levels_listed : int, optional
        Maximum number of levels to list in diagnostic output. Default 30.
    compute_level_counts : bool, optional
        Whether to compute non-null counts per level (can be expensive for
        large Dask arrays). Default ``True``.

    Returns
    -------
    dict[str, Any]
        Summary with keys ``"group"``, ``"src_dim_sizes"``, ``"sub_dim_sizes"``,
        ``"scalar_dim_conflicts_src"``, ``"scalar_dim_conflicts_sub"``,
        ``"horizontal_dims"``, ``"vertical_dims"``,
        ``"vars_missing_vertical_dim"``, ``"levels_all_nan"``, and ``"notes"``.
    """
    summary: dict[str, Any] = {
        "group": str(group_key),
        "src_dim_sizes": dict(ds_src.sizes),
        "sub_dim_sizes": dict(ds_sub.sizes),
        "scalar_dim_conflicts_src": _dataset_scalar_dim_conflicts(ds_src),
        "scalar_dim_conflicts_sub": _dataset_scalar_dim_conflicts(ds_sub),
        "horizontal_dims": (),
        "vertical_dims": (),
        "vars_missing_vertical_dim": [],
        "levels_all_nan": {},
        "notes": [],
    }

    _log(logger, "info", f"[polar_diag] group={group_key!r}")
    _log(logger, "info", f"[polar_diag]   ds_src sizes: {_fmt_dim_sizes(ds_src)}")
    _log(logger, "info", f"[polar_diag]   ds_sub sizes: {_fmt_dim_sizes(ds_sub)}")

    if summary["scalar_dim_conflicts_src"]:
        _log(
            logger,
            "warning",
            f"[polar_diag]   ds_src scalar-dim conflicts: {summary['scalar_dim_conflicts_src']}",
        )
    if summary["scalar_dim_conflicts_sub"]:
        _log(
            logger,
            "warning",
            f"[polar_diag]   ds_sub scalar-dim conflicts: {summary['scalar_dim_conflicts_sub']}",
        )

    h_dims, h_src = _horizontal_dims_from_lonlat(ds_src, lon_name=lon_name, lat_name=lat_name)
    summary["horizontal_dims"] = h_dims
    _log(logger, "info", f"[polar_diag]   horizontal dims: {h_dims} (source={h_src})")

    cand_v = [
        d
        for d in ds_sub.dims
        if d != time_dim and d not in h_dims and d not in ("r_km", "theta_deg")
    ]
    if group_key in ds_sub.dims and group_key != time_dim:
        v_dims = [str(group_key)] + [d for d in cand_v if d != group_key]
    else:
        v_dims = cand_v

    summary["vertical_dims"] = tuple(v_dims)
    _log(logger, "info", f"[polar_diag]   vertical dims candidates: {summary['vertical_dims']}")

    for cname in (lon_name, lat_name):
        if cname in ds_src:
            _log(
                logger,
                "info",
                f"[polar_diag]   src coord {cname!r}: {_describe_coord(ds_src[cname])}",
            )
        else:
            _log(logger, "warning", f"[polar_diag]   src missing coord {cname!r}")

    for vd in v_dims[:3]:
        if vd in ds_sub.coords:
            _log(
                logger,
                "info",
                f"[polar_diag]   sub coord {vd!r}: "
                f"{_describe_coord(ds_sub.coords[vd], max_listed=max_levels_listed)}",
            )
        elif vd in ds_sub.variables:
            _log(
                logger,
                "info",
                f"[polar_diag]   sub var {vd!r}: "
                f"{_describe_coord(ds_sub[vd], max_listed=max_levels_listed)}",
            )
        else:
            _log(logger, "warning", f"[polar_diag]   sub missing vertical coord/var {vd!r}")

    v_main = str(group_key) if str(group_key) in ds_sub.dims else (v_dims[0] if v_dims else None)
    if v_main is not None:
        missing_v = []
        for vname, da in ds_sub.data_vars.items():
            if vname in (lon_name, lat_name):
                continue
            if v_main in ds_sub.dims and v_main not in da.dims:
                missing_v.append(vname)

        summary["vars_missing_vertical_dim"] = missing_v
        if missing_v:
            _log(
                logger,
                "warning",
                f"[polar_diag]   vars missing vertical dim {v_main!r}: {missing_v[:max_vars]}",
            )

    if compute_level_counts and v_main is not None and v_main in ds_sub.dims and h_dims:
        for vname, da in list(ds_sub.data_vars.items())[:max_vars]:
            if v_main not in da.dims:
                continue
            try:
                red_dims = tuple(d for d in h_dims if d in da.dims)
                if time_dim in da.dims:
                    red_dims = (time_dim,) + red_dims

                cnt = da.notnull().sum(dim=red_dims)
                cntv = cnt.compute() if hasattr(cnt.data, "compute") else cnt
                bad = np.asarray(cntv.values == 0)

                if bad.any():
                    if v_main in ds_sub.coords:
                        levels = ds_sub[v_main].values
                    else:
                        levels = np.arange(ds_sub.sizes[v_main])
                    bad_levels = np.asarray(levels)[bad]
                    summary["levels_all_nan"][vname] = bad_levels[:max_levels_listed].tolist()

            except Exception as exc:
                summary["notes"].append(f"level_nan_check_failed:{vname}:{type(exc).__name__}")

        if summary["levels_all_nan"]:
            _log(
                logger,
                "warning",
                f"[polar_diag]   variables with all-NaN levels"
                f" (up to {max_levels_listed} levels per var):",
            )
            for vname, lev in summary["levels_all_nan"].items():
                _log(logger, "warning", f"[polar_diag]     - {vname}: {lev}")
    else:
        summary["notes"].append("level_nan_check_skipped")

    return summary


def polar_select_and_diagnose(
    dico_collocated: dict[str, xr.Dataset],
    *,
    conf: Any,
    logger: logging.Logger | None = None,
    max_vars: int = 50,
    max_levels_listed: int = 30,
    compute_level_counts: bool = True,
) -> tuple[dict[str, xr.Dataset], dict[str, Any]]:
    """
    Select polar-projection inputs and emit vertical-coordinate diagnostics.

    Performs the variable-selection step of polar projection without calling
    xESMF.  Useful for debugging AROME vertical-coordinate inconsistencies
    before running the full regridding.

    Parameters
    ----------
    dico_collocated : dict[str, xr.Dataset]
        Per-group collocated datasets keyed by vertical group name.
    conf : object
        Configuration object.  Reads ``compute_polar_proj``, ``name_longitude``,
        ``name_latitude``, ``polar_variables``, and polar-grid sizing attributes.
    logger : logging.Logger or None, optional
        Logger for diagnostic messages.  When ``None``, messages are printed.
        Default ``None``.
    max_vars : int, optional
        Maximum number of variables to inspect per group. Default 50.
    max_levels_listed : int, optional
        Maximum number of levels to list in diagnostic output. Default 30.
    compute_level_counts : bool, optional
        Whether to compute non-null counts per level. Default ``True``.

    Returns
    -------
    dico_selected : dict[str, xr.Dataset]
        Per-group selected datasets, keyed by output group name.  Empty when
        ``conf.compute_polar_proj`` is ``False`` or all requests are skipped.
    report : dict[str, Any]
        Report with keys ``"skipped"``, ``"redirects"``, ``"selected"``,
        ``"diagnostics"``, ``"n_groups_out"``, and ``"n_groups_requested"``.
    """
    compute = bool(getattr(conf, "compute_polar_proj", False))
    if not compute:
        return {}, {"skipped": [], "notes": ["compute_polar_proj is False"]}

    time_dim = "time"
    lon_name = _get_required(conf, "name_longitude")
    lat_name = _get_required(conf, "name_latitude")

    reqs = _get_polar_requests(conf)
    if not reqs:
        return {}, {"skipped": [], "notes": ["No polar_variables request found."]}

    if "surface" in dico_collocated:
        ds_ref = dico_collocated["surface"]
    else:
        ds_ref = next(iter(dico_collocated.values()))

    try:
        grid = PolarLonLatGrid.from_conf(conf=conf)
        _ = grid.build(ds_ref)
    except Exception as exc:
        _log(
            logger, "warning", f"[polar_diag] polar grid build failed: {type(exc).__name__}: {exc}"
        )

    report: dict[str, Any] = {"skipped": [], "redirects": [], "selected": [], "diagnostics": []}
    dico_selected: dict[str, xr.Dataset] = {}

    for group_key, req in reqs.items():
        group_key = str(group_key)
        out_key = "level" if group_key == "level_w" else group_key

        src_key = group_key
        if group_key == "level_w":
            if "level" in dico_collocated:
                src_key = "level"
                report["redirects"].append({"from": "level_w", "to": "level"})
            elif "level_w" in dico_collocated:
                src_key = "level_w"
                report["redirects"].append({"from": "level_w", "to": "level_w"})
            else:
                report["skipped"].append(
                    {"group": group_key, "reason": "missing group level and level_w"}
                )
                continue
        else:
            if src_key not in dico_collocated:
                report["skipped"].append({"group": group_key, "reason": "missing group"})
                continue

        ds_src = dico_collocated[src_key]
        sel_group = src_key if group_key == "level_w" else group_key

        ds_sub = build_group_ds(
            ds=ds_src,
            req=req,
            group_key=sel_group,
            concat_dimension=time_dim,
            float_tol=None,
        )
        if ds_sub is None or len(ds_sub.data_vars) == 0:
            report["skipped"].append({"group": group_key, "reason": "no matching variables"})
            continue

        diag = _diagnose_group_datasets(
            group_key=group_key,
            ds_src=ds_src,
            ds_sub=ds_sub,
            lon_name=lon_name,
            lat_name=lat_name,
            time_dim=time_dim,
            logger=logger,
            max_vars=max_vars,
            max_levels_listed=max_levels_listed,
            compute_level_counts=compute_level_counts,
        )
        report["diagnostics"].append(diag)

        dico_selected[out_key] = ds_sub
        report["selected"].append(
            {
                "requested_group": group_key,
                "source_group": src_key,
                "output_group": out_key,
                "nvars": len(ds_sub.data_vars),
            }
        )

    report["n_groups_out"] = len(dico_selected)
    report["n_groups_requested"] = len(reqs)
    return dico_selected, report
