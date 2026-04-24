# Copyright 2026 Clément Soufflet, Météo-France
# Licensed under the Apache License, Version 2.0
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def add_vrad_vtan_from_polar_dict(
    dict_polar_user: dict[str, xr.Dataset], conf
) -> dict[str, xr.Dataset]:
    """
    Add vrad and vtan to the relevant polar dataset in dict_polar_user, depending on model.
    Only for variables on model level

    Assumptions
    -----------
    - theta is in radians and stored as coordinate "theta" with dimension "theta".
    - u is eastward, v is northward.
    - vrad is positive outward.
    - vtan is positive for clockwise rotation (theta increases clockwise from North).
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


def add_speed_from_uv_dict(
    dict_user: dict[str, xr.Dataset],
    conf,
    *,
    speed_name: str = "wind_speed",
) -> dict[str, xr.Dataset]:
    """
    Compute and add horizontal wind speed to the relevant group in ``dict_user``.

    Parameters
    ----------
    dict_user : dict[str, xr.Dataset]
        User datasets keyed by vertical group.  The target group is
        "heightAboveGround" for AROME and "level" for MNH.
    conf : object
        Configuration object.  Reads ``atm_model`` (selects the target group)
        and ``velocity_aliases`` (keys "u_velocity" and "v_velocity").
    speed_name : str, optional
        Name under which the speed variable is stored. Default "wind_speed".

    Returns
    -------
    dict[str, xr.Dataset]
        Updated dictionary with ``speed_name = hypot(u, v)`` added to the
        target group.  Returns ``dict_user`` unchanged when the group or
        required velocity variables are missing.
    """
    group = "heightAboveGround" if conf.atm_model == "AROME" else "level"

    if group not in dict_user:
        logger.info("Speed not computed: '%s' group not found in input dict.", group)
        return dict_user

    ds = dict_user[group]

    u_name = conf.velocity_aliases["u_velocity"]
    v_name = conf.velocity_aliases["v_velocity"]

    if (u_name not in ds) or (v_name not in ds):
        logger.info(
            "Speed not computed in group '%s': missing variables (u='%s', v='%s').",
            group,
            u_name,
            v_name,
        )
        return dict_user

    u = ds[u_name]
    v = ds[v_name]

    speed = np.hypot(u, v)

    units = u.attrs.get("units", None)
    speed.attrs.update(
        {
            "long_name": "horizontal wind speed",
            "units": units,
        }
    )

    dict_user[group] = ds.assign({speed_name: speed})
    return dict_user
