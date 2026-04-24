# Copyright 2026 Clément Soufflet, Météo-France
# Licensed under the Apache License, Version 2.0
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ----------------------------
# Core numerics (shift mean)
# ----------------------------
def _pairmean(a: xr.DataArray, b: xr.DataArray, policy: str) -> xr.DataArray:
    if policy == "strict":
        ok = np.isfinite(a) & np.isfinite(b)
        return xr.where(ok, 0.5 * (a + b), np.nan)

    # partial
    wa = xr.where(np.isfinite(a), 1.0, 0.0)
    wb = xr.where(np.isfinite(b), 1.0, 0.0)
    num = xr.where(wa > 0, a, 0.0) + xr.where(wb > 0, b, 0.0)
    den = wa + wb
    return xr.where(den > 0, num / den, np.nan)


def _collocate_keepN(da: xr.DataArray, dim: str, use_forward, policy: str) -> xr.DataArray:
    """
    Stagger-average along ``dim`` while keeping the same length.

    Parameters
    ----------
    da : xr.DataArray
        Input field to collocate.
    dim : str
        Dimension along which stagger-averaging is applied.
    use_forward : bool or xr.DataArray
        If True, use forward pairing ``mean(i, i+1)``; otherwise backward
        ``mean(i-1, i)``.
    policy : {"partial", "strict"}
        Edge handling: "partial" keeps the single available value at the
        boundary; "strict" produces NaN.

    Returns
    -------
    xr.DataArray
        Collocated field with the same shape as ``da``.
    """
    fwd = _pairmean(da, da.shift({dim: -1}), policy)
    bwd = _pairmean(da.shift({dim: 1}), da, policy)
    return xr.where(use_forward, fwd, bwd)


def _dir_from_coords(c_center: xr.DataArray, c_stag: xr.DataArray, dim_box: str) -> xr.DataArray:
    """
    Infer the stagger-averaging direction from center and staggered coordinate arrays.

    Parameters
    ----------
    c_center : xr.DataArray
        Coordinate values at mass-point centers.
    c_stag : xr.DataArray
        Coordinate values at staggered (edge) points.
    dim_box : str
        Box dimension over which the median offset is computed.

    Returns
    -------
    xr.DataArray
        Boolean DataArray (typically on the "time" dimension).
        True means forward pairing (center > stag on average).
    """
    d = c_center - c_stag
    if dim_box in d.dims:
        d = d.median(dim_box, skipna=True)
    return d > 0


def _get_uv_box_dims(da: xr.DataArray) -> tuple[str, str]:
    """
    Return the last two dimension names of a wind field (y_box, x_box).

    Parameters
    ----------
    da : xr.DataArray
        At least 2D wind component field.

    Returns
    -------
    tuple[str, str]
        ``(y_dim, x_dim)`` — last two dimension names.

    Raises
    ------
    ValueError
        If ``da`` has fewer than 2 dimensions.
    """
    if da.ndim < 2:
        raise ValueError("Expected at least 2D field for horizontal collocation.")
    return da.dims[-2], da.dims[-1]  # (y_box, x_box) in your extraction


# ----------------------------
# W vertical collocation
# ----------------------------
def _collocate_w_to_zC(
    w: xr.DataArray,
    *,
    zC_name: str,
    zC_coord: xr.DataArray,
    policy: str,
) -> xr.DataArray:
    """
    Collocate the W (vertical velocity) field onto the mass-point vertical grid.

    Parameters
    ----------
    w : xr.DataArray
        Vertical velocity field.  The third-from-last dimension (index -3)
        is treated as the native vertical dimension.
    zC_name : str
        Name of the target mass-point vertical dimension.
    zC_coord : xr.DataArray
        Coordinate values of the target vertical dimension.
    policy : {"partial", "strict"}
        Edge handling for stagger-averaging.

    Returns
    -------
    xr.DataArray
        W field on the ``zC_name`` vertical grid with ``len(zC_coord)`` levels.

    Notes
    -----
    Four cases are handled in order:

    1. Source dim == ``zC_name``: no-op (coordinates assigned only).
    2. ``nzW == nzC``: keep-N stagger-mean.
    3. ``nzW == nzC + 1``: true interface-to-center averaging.
    4. Otherwise: keep-N then interpolate to ``zC_coord`` if coordinates allow.
    """
    # If not enough dims to have a vertical dimension, return as-is
    if w.ndim < 3:
        logger.debug(
            "W collocation skipped, w.ndim=%s < 3, dims=%s.",
            w.ndim,
            w.dims,
        )
        return w

    z_w = w.dims[-3]
    if z_w == zC_name:
        logger.debug(
            "W already on target vertical dim, z_w=%s, zC_name=%s. Assigning coords only.",
            z_w,
            zC_name,
        )
        return w.assign_coords({zC_name: zC_coord})

    zW_coord = w.coords.get(z_w, None)
    nzC = zC_coord.sizes[zC_name]
    nzW = w.sizes[z_w]

    logger.debug(
        "Vertical collocation of W, source dim=%s (nzW=%s) to target dim=%s (nzC=%s), policy=%s.",
        z_w,
        nzW,
        zC_name,
        nzC,
        policy,
    )

    # Direction inference (forward/backward) if 1D coords exist
    if (zW_coord is not None) and (zW_coord.ndim == 1) and (zW_coord.sizes[z_w] == nzW):
        nmin = min(nzC, nzW)
        use_fwd_z = float(np.nanmedian(zC_coord.values[:nmin] - zW_coord.values[:nmin])) > 0
        logger.debug(
            "Vertical pairing direction inferred from coords, use_fwd_z=%s (nmin=%s).",
            use_fwd_z,
            nmin,
        )
    else:
        use_fwd_z = True
        logger.debug(
            "Vertical pairing direction defaulted to forward "
            "(missing or incompatible 1D coord for %s).",
            z_w,
        )

    if nzW == nzC:
        logger.debug("Case nzW == nzC, applying keep-N shift-mean on %s.", z_w)
        wC = _collocate_keepN(w, z_w, use_fwd_z, policy)
        wC = wC.rename({z_w: zC_name}).assign_coords({zC_name: zC_coord})
        return wC

    if nzW == nzC + 1:
        logger.debug("Case nzW == nzC + 1, averaging interfaces to centers on %s.", z_w)
        wC = _pairmean(w, w.shift({z_w: -1}), policy).isel({z_w: slice(0, -1)})
        wC = wC.rename({z_w: zC_name}).assign_coords({zC_name: zC_coord})
        return wC

    # Fallback: collocate on z_w then interpolate to zC_coord
    logger.debug(
        "Fallback case for W vertical sizes, nzW=%s, nzC=%s."
        " Collocate on %s then interpolate if possible.",
        nzW,
        nzC,
        z_w,
    )
    w_tmp = _collocate_keepN(w, z_w, use_fwd_z, policy)
    if zW_coord is not None:
        w_tmp = w_tmp.assign_coords({z_w: zW_coord})
        w_i = w_tmp.interp({z_w: zC_coord}, kwargs={"fill_value": np.nan})
        w_i = w_i.rename({z_w: zC_name}).assign_coords({zC_name: zC_coord})
        logger.debug("Interpolation from %s to %s completed.", z_w, zC_name)
        return w_i

    logger.error(
        "Incompatible vertical sizes and no coord to interpolate,"
        " z_w=%s (nzW=%s), zC_name=%s (nzC=%s).",
        z_w,
        nzW,
        zC_name,
        nzC,
    )
    raise ValueError(
        f"Incompatible vertical sizes and no coord to interpolate: {z_w}={nzW}, {zC_name}={nzC}."
    )


# ----------------------------
# Main entry point for dico_user
# ----------------------------
def collocate_winds(
    dico_user: dict[str, xr.Dataset],
    *,
    conf,
    policy: str = "partial",  # "partial" | "strict"
    drop_level_w_group: bool = True,  # MNH only, to avoid confusion
) -> dict[str, xr.Dataset]:
    """
    Post-extraction wind collocation for MNH (no-op for AROME).

    U and V are collocated from their staggered positions to the mass-point
    grid.  W is collocated vertically onto the mass-point vertical grid and
    injected into the main group, replacing the separate level_w group.

    Parameters
    ----------
    dico_user : dict[str, xr.Dataset]
        Extracted user datasets keyed by vertical group (e.g. "level",
        "level_w", "surface").
    conf : object
        Configuration object.  Reads ``atm_model``, ``name_vertical_dim``,
        and ``velocity_aliases``.
    policy : {"partial", "strict"}, optional
        Stagger-averaging edge policy. Default "partial".
    drop_level_w_group : bool, optional
        When True, remove the separate "level_w" group after injecting WT into
        the main group (MNH only). Default True.

    Returns
    -------
    dict[str, xr.Dataset]
        Updated dictionary where wind variables in the main group have been
        replaced by their collocated versions.  The level_w key is removed
        when ``drop_level_w_group=True`` and the model is MNH.

    Raises
    ------
    ValueError
        If ``policy`` is not "partial" or "strict", or if the main group is
        missing its vertical coordinate.
    """
    if policy not in {"partial", "strict"}:
        raise ValueError("policy must be 'partial' or 'strict'.")

    atm_model = str(conf.atm_model).upper()
    zC_name = conf.name_vertical_dim  # also used as dict key (e.g., "level")

    logger.debug(
        "collocate_winds called, atm_model=%s, zC_name=%s, policy=%s,"
        " drop_level_w_group=%s, dico_keys=%s.",
        atm_model,
        zC_name,
        policy,
        drop_level_w_group,
        list(dico_user.keys()),
    )

    if zC_name not in dico_user:
        logger.debug("Main vertical group %r not found in dico_user, no-op.", zC_name)
        return dico_user

    ds_h = dico_user[zC_name].copy(deep=False)
    out = ds_h.copy(deep=False)

    # Wind variable names from conf (e.g., UT/VT/WT for MNH, u/v/w for other models)
    vel = getattr(conf, "velocity_aliases", {}) or {}
    u_name = vel.get("u_velocity", "u")
    v_name = vel.get("v_velocity", "v")
    w_name = vel.get("w_velocity", "w")

    logger.debug(
        "Velocity aliases resolved, u_name=%s, v_name=%s, w_name=%s.",
        u_name,
        v_name,
        w_name,
    )

    # Identify where W lives (same group or separate group)
    ds_w = None
    key_w = None
    if w_name in out.data_vars:
        ds_w = out
        key_w = zC_name
        logger.debug("W found in main group %r as variable %r.", zC_name, w_name)
    else:
        for k, ds in dico_user.items():
            if (w_name in ds.data_vars) and ("level_w" in ds.coords):
                ds_w = ds
                key_w = k
                logger.debug(
                    "W found in separate group %r as variable %r (coord 'level_w' present).",
                    k,
                    w_name,
                )
                break
        if ds_w is None:
            logger.debug("W variable %r not found in any group with coord 'level_w'.", w_name)

    # Vertical coordinate must exist in main group for WT insertion
    if zC_name not in out.coords:
        logger.error("Main 3D group is missing vertical coordinate %r.", zC_name)
        raise ValueError(f"Main 3D group is missing vertical coordinate {zC_name!r}.")

    # ----------------------------
    # U, V (overwrite with collocated versions, keep original names)
    # ----------------------------
    if (u_name in out.data_vars) and (v_name in out.data_vars):
        u = out[u_name]
        v = out[v_name]

        y_dim_u, x_dim_u = _get_uv_box_dims(u)
        y_dim_v, x_dim_v = _get_uv_box_dims(v)

        logger.debug(
            "U/V present. U dims=%s (colloc along x_dim=%s), V dims=%s (colloc along y_dim=%s).",
            u.dims,
            x_dim_u,
            v.dims,
            y_dim_v,
        )

        if atm_model == "MNH":
            # Use ni/ni_u and nj/nj_v if present, otherwise default forward
            xC = getattr(conf, "name_lon_dim", "ni")
            yC = getattr(conf, "name_lat_dim", "nj")
            xU = f"{xC}_u"
            yV = f"{yC}_v"

            use_fwd_x = True
            use_fwd_y = True

            if (xC in out.coords) and (xU in out.coords) and (x_dim_u in out.coords[xC].dims):
                use_fwd_x = _dir_from_coords(out.coords[xC], out.coords[xU], x_dim_u)
                logger.debug(
                    "MNH U x-direction inferred using coords %s/%s on dim_box=%s.",
                    xC,
                    xU,
                    x_dim_u,
                )
            else:
                logger.debug(
                    "MNH U x-direction defaulted to forward"
                    " (missing coords %s or %s, or incompatible dims).",
                    xC,
                    xU,
                )

            if (yC in out.coords) and (yV in out.coords) and (y_dim_v in out.coords[yC].dims):
                use_fwd_y = _dir_from_coords(out.coords[yC], out.coords[yV], y_dim_v)
                logger.debug(
                    "MNH V y-direction inferred using coords %s/%s on dim_box=%s.",
                    yC,
                    yV,
                    y_dim_v,
                )
            else:
                logger.debug(
                    "MNH V y-direction defaulted to forward"
                    " (missing coords %s or %s, or incompatible dims).",
                    yC,
                    yV,
                )

            out[u_name] = _collocate_keepN(u, x_dim_u, use_fwd_x, policy).rename(u_name)
            out[v_name] = _collocate_keepN(v, y_dim_v, use_fwd_y, policy).rename(v_name)

            logger.debug("U/V collocation applied for MNH, policy=%s.", policy)
        else:
            # AROME (or others): assumed already on mass point in your workflow (no-op)
            out[u_name] = u.rename(u_name)
            out[v_name] = v.rename(v_name)
            logger.debug("U/V collocation no-op for atm_model=%s.", atm_model)
    else:
        logger.debug(
            "U/V collocation skipped, missing variables. u_present=%s, v_present=%s.",
            (u_name in out.data_vars),
            (v_name in out.data_vars),
        )

    # ----------------------------
    # W (overwrite with collocated version, keep original name)
    # ----------------------------
    if (ds_w is not None) and (w_name in ds_w.data_vars):
        w = ds_w[w_name]
        logger.debug(
            "W collocation starting, source_group=%r, w_name=%r, w.dims=%s.",
            key_w,
            w_name,
            w.dims,
        )

        wC = _collocate_w_to_zC(
            w,
            zC_name=zC_name,
            zC_coord=out.coords[zC_name],
            policy=policy,
        )
        out[w_name] = wC.rename(w_name)

        logger.debug("W collocation completed, new dims=%s.", out[w_name].dims)

        # Ensure WT shares the same horizontal coords as the main group, when possible
        for cname in ("x_box", "y_box", "x_box_km", "y_box_km", "latitude", "longitude"):
            if cname in out.coords and cname not in out[w_name].coords:
                out[w_name] = out[w_name].assign_coords({cname: out.coords[cname]})
                logger.debug("Assigned missing coord %r to W from main group.", cname)
    else:
        logger.debug("W collocation skipped, W not available or not identified.")

    # Build output dict with only the main group replaced
    dico_out = dict(dico_user)
    dico_out[zC_name] = out

    # Optionally drop the separate level_w group (MNH only)
    if drop_level_w_group and atm_model == "MNH" and key_w is not None and key_w != zC_name:
        dico_out.pop(key_w, None)
        logger.info(
            "Dropped separate W group %r (WT injected into main group %r).",
            key_w,
            zC_name,
        )

    logger.debug("collocate_winds finished, output keys=%s.", list(dico_out.keys()))
    return dico_out
