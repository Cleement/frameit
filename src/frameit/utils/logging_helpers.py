# Copyright 2026 Clément Soufflet, Météo-France
# Licensed under the Apache License, Version 2.0
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_global_params_and_presets(conf, pattern: str) -> None:
    """
    Log model names, the file glob pattern, and model preset keys at INFO level.

    Parameters
    ----------
    conf : SimulationConfig
        Loaded simulation configuration.
    pattern : str
        Glob pattern used to find input files (e.g. "MNH_file???.nc").
    """
    logger.info("Atmospheric model: %s", conf.atm_model)
    logger.info("Ocean model: %s", conf.ocean_model)
    logger.info("Wave model: %s", conf.wave_model)
    logger.info("Files pattern: %s", pattern)

    preset_fields = getattr(conf, "simulation_parameter", {}).get("preset_fields_coords_dims", {})
    if preset_fields:
        logger.info("Model preset keys for %s:", conf.atm_model)
        for k, v in preset_fields.items():
            logger.info("\t\t\t- %s: %s", k, v, extra={"noprefix": True})


def log_user_requested_vars(conf) -> None:
    """
    Log user-requested variable groups and their level-selection details.

    Parameters
    ----------
    conf : SimulationConfig
        Loaded simulation configuration.  Reads ``requested_variables_user``.
    """
    user_req = getattr(conf, "requested_variables_user", {}) or {}
    if not user_req:
        logger.info("User requested variables: <none>")
        return

    logger.info("User requested variables:")
    for g, spec in user_req.items():
        spec = spec or {}
        vars_ = spec.get("variables", [])
        mode = (spec.get("level_selection") or "").lower()
        if mode == "values":
            extra_sel = f" (levels={spec.get('level_values', [])})"
        elif mode == "indices":
            extra_sel = f" (indices={spec.get('level_indices', [])})"
        elif mode == "all":
            extra_sel = " (all levels)"
        else:
            extra_sel = ""
        logger.info("\t\t\t- %s: %s%s", g, vars_, extra_sel, extra={"noprefix": True})


def log_tracker_requested_vars(conf) -> None:
    """
    Log tracker-requested variable groups for the active tracking method.

    Parameters
    ----------
    conf : SimulationConfig
        Loaded simulation configuration.  Reads ``requested_variables_tracker``
        and ``tracking_method``.
    """
    trk_all = getattr(conf, "requested_variables_tracker", None)
    if trk_all is None:
        trk_all = getattr(conf, "requested_vars_tracker", {}) or {}

    method = getattr(conf, "tracking_method", None) or "wind_pressure"
    trk_req = trk_all.get(method, {}) if isinstance(trk_all, dict) else {}

    if not trk_req:
        logger.info(
            "No tracker variables found for method '%s'",
            method,
            extra={"noprefix": True},
        )
        return

    logger.info("Tracker '%s' requested variables:", method)
    for g, spec in trk_req.items():
        spec = spec or {}
        vars_ = spec.get("variables", [])
        mode = (spec.get("level_selection") or "").lower()
        if mode == "values":
            extra_sel = f" (levels={spec.get('level_values', [])})"
        elif mode == "indices":
            extra_sel = f" (indices={spec.get('level_indices', [])})"
        elif mode == "all":
            extra_sel = " (all levels)"
        else:
            extra_sel = ""
        logger.info("\t\t\t- %s: %s%s", g, vars_, extra_sel, extra={"noprefix": True})
