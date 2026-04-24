# Copyright 2026 Clément Soufflet, Météo-France
# Licensed under the Apache License, Version 2.0
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def ensure_theta_rad_dim(ds: xr.Dataset, *, theta_deg: str = "theta_deg") -> xr.Dataset:
    """
    Replace the ``theta_deg`` dimension with a radian ``theta`` dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``theta_deg`` dimension whose coordinate values are
        azimuths in degrees.
    theta_deg : str, optional
        Name of the azimuth dimension in degrees. Default ``"theta_deg"``.

    Returns
    -------
    xr.Dataset
        Dataset with the azimuthal dimension renamed to ``"theta"`` (radians),
        a ``"theta_deg"`` coordinate kept for reference, and ``units``/
        ``long_name`` attributes set on both coordinates.  Returns ``ds``
        unchanged when ``theta_deg`` is absent.
    """
    if theta_deg not in ds.dims:
        logger.debug("ensure_theta_rad_dim: no-op (missing dim %r).", theta_deg)
        return ds

    theta_deg_vals = ds[theta_deg].values.astype(float)
    theta_rad_vals = np.deg2rad(theta_deg_vals)

    logger.debug(
        "ensure_theta_rad_dim: converting dim %r -> 'theta' (n=%d).",
        theta_deg,
        int(theta_deg_vals.size),
    )

    ds = ds.assign_coords(theta=(theta_deg, theta_rad_vals))
    ds = ds.swap_dims({theta_deg: "theta"})

    # After swap, keep an explicit theta_deg coord on theta
    if "theta_deg" in ds.coords and ds["theta_deg"].dims != ("theta",):
        ds = ds.assign_coords(theta_deg=("theta", theta_deg_vals))
        logger.debug("ensure_theta_rad_dim: re-assigned coord 'theta_deg' on dim 'theta'.")

    # Attributes (do not change data)
    ds["theta"].attrs.update(units="rad", long_name="geodesic azimuth (clockwise from north)")
    ds["theta_deg"].attrs.update(
        units="degree", long_name="geodesic azimuth (clockwise from north)"
    )

    if "rr" in ds.coords or "rr" in ds.variables:
        ds["rr"].attrs.update(units="km", long_name="radial distance from center")
    else:
        logger.debug("ensure_theta_rad_dim: variable/coord 'r' not found, skipping r attrs update.")

    return ds


def close_theta(ds: xr.Dataset, *, theta_dim: str = "theta") -> xr.Dataset:
    """
    Close the azimuthal axis by appending ``theta0 + 2π`` as a duplicate first column.

    Intended for plotting: contour and pcolor renderers need the full 360° wrap.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``theta_dim`` dimension in radians.  When the dimension
        is absent the function returns ``ds`` unchanged.
    theta_dim : str, optional
        Name of the azimuthal dimension. Default ``"theta"``.

    Returns
    -------
    xr.Dataset
        Dataset with one extra point appended along ``theta_dim``, equal to the
        first slice at ``theta = theta0 + 2π``.  The ``theta_closed`` attribute
        is set to ``True``.
    """
    if theta_dim not in ds.dims:
        logger.debug("close_theta: no-op (missing dim %r).", theta_dim)
        return ds

    n0 = int(ds.sizes.get(theta_dim, 0))
    theta0 = float(ds[theta_dim].isel({theta_dim: 0}).values)
    theta_end = theta0 + 2.0 * np.pi

    logger.debug(
        "close_theta: closing %r (n=%d) by appending theta_end=theta0+2*pi"
        " (theta0=%.6f, theta_end=%.6f).",
        theta_dim,
        n0,
        theta0,
        theta_end,
    )

    tail = ds.isel({theta_dim: 0}).expand_dims({theta_dim: [theta_end]})

    # If theta_deg exists along theta_dim, close it consistently
    if "theta_deg" in ds.coords and "theta" in ds.dims and ds["theta_deg"].dims == (theta_dim,):
        deg0 = float(ds["theta_deg"].isel({theta_dim: 0}).values)
        tail = tail.assign_coords(theta_deg=(theta_dim, [deg0 + 360.0]))
        logger.debug(
            "close_theta: closed theta_deg consistently (deg0=%.3f -> %.3f).", deg0, deg0 + 360.0
        )

    ds_closed = xr.concat([ds, tail], dim=theta_dim)

    ds_closed.attrs["theta_closed"] = True
    logger.debug(
        "close_theta: done. new size=%d (was %d).",
        int(ds_closed.sizes.get(theta_dim, 0)),
        n0,
    )
    return ds_closed


def finalize_polar_output(ds: xr.Dataset) -> xr.Dataset:
    """
    Apply FrameIt geodesic conventions to a raw polar output dataset.

    Calls :func:`ensure_theta_rad_dim` then :func:`close_theta` and stamps
    global attributes describing the theta convention.

    Parameters
    ----------
    ds : xr.Dataset
        Raw polar dataset produced by the projection step, with a
        ``"theta_deg"`` dimension.

    Returns
    -------
    xr.Dataset
        Dataset with dim ``"theta"`` (radians), coord ``"theta_deg"``, closed
        azimuthal axis, and attrs ``theta_reference``, ``theta_zero_location``,
        ``theta_direction``, ``theta_closed``.
    """
    logger.debug(
        "finalize_polar_output: start. dims=%s, coords=%s, data_vars=%s.",
        tuple(ds.dims),
        tuple(ds.coords),
        tuple(ds.data_vars),
    )

    ds = ensure_theta_rad_dim(ds)
    ds = close_theta(ds, theta_dim="theta")

    ds.attrs.update(
        theta_reference="geodesic",
        theta_zero_location="north",
        theta_direction="clockwise",
        theta_closed=True,
    )

    logger.debug("finalize_polar_output: done. dims=%s.", tuple(ds.dims))
    return ds


def add_vrad_vtan_from_polar_dict(
    dict_polar_user: dict[str, xr.Dataset], conf
) -> dict[str, xr.Dataset]:
    """
    Compute radial and tangential wind components and add them to the polar dataset.

    Parameters
    ----------
    dict_polar_user : dict[str, xr.Dataset]
        Per-group polar datasets.  The target group is ``"heightAboveGround"``
        for AROME and ``"level"`` for MNH.
    conf : object
        Configuration object.  Reads ``atm_model`` and ``velocity_aliases``
        (keys ``"u_velocity"`` and ``"v_velocity"``).

    Returns
    -------
    dict[str, xr.Dataset]
        Updated dictionary with ``vrad`` and ``vtan`` added to the target group.
        Returns ``dict_polar_user`` unchanged when the group or required velocity
        variables are absent, or when the ``"theta"`` coordinate (radians) is
        missing.

    Notes
    -----
    - ``theta`` must be in radians, stored as coordinate ``"theta"`` with
      dimension ``"theta"``.
    - ``u`` is eastward wind, ``v`` is northward wind.
    - ``vrad`` is positive outward from the storm center.
    - ``vtan`` is positive for clockwise rotation (theta increases clockwise
      from North).
    """
    group = "heightAboveGround" if conf.atm_model == "AROME" else "level"

    if group not in dict_polar_user:
        return dict_polar_user

    ds = dict_polar_user[group]

    u_name = conf.velocity_aliases["u_velocity"]
    v_name = conf.velocity_aliases["v_velocity"]

    if (u_name not in ds) or (v_name not in ds):
        logger.info("vrad/vtan not computed: '%s' group not found in dict_polar_user.", group)
        return dict_polar_user

    if "theta" not in ds.coords:
        logger.info("vrad/vtan not computed: missing 'theta' coordinate (radians).")
        return dict_polar_user

    u = ds[u_name]
    v = ds[v_name]
    theta = ds["theta"]  # radians

    sin_t = xr.apply_ufunc(np.sin, theta)
    cos_t = xr.apply_ufunc(np.cos, theta)

    vrad = u * sin_t + v * cos_t
    vtan = u * cos_t - v * sin_t

    units = u.attrs.get("units", None)
    vrad.attrs.update(
        {"long_name": "radial wind component (positive outward from storm center)", "units": units}
    )
    vtan.attrs.update(
        {"long_name": "tangential wind component (positive for clockwise rotation)", "units": units}
    )

    dict_polar_user[group] = ds.assign(vrad=vrad, vtan=vtan)

    return dict_polar_user
