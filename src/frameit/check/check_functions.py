# Copyright 2026 Clément Soufflet, Météo-France
# Licensed under the Apache License, Version 2.0
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def drop_time_dupes(ds: xr.Dataset, keep: str = "first") -> xr.Dataset:
    """
    Remove duplicate time steps from a Dataset, keeping one occurrence per step.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a "time" dimension.
    keep : {"first", "last"}, optional
        Which occurrence to keep when duplicates are present. Default "first".

    Returns
    -------
    xr.Dataset
        Dataset with unique time steps, preserving ascending time order.
    """
    # keep: "first" or "last"
    t = ds["time"].values
    if keep == "last":
        # indices of the last occurrences
        _, idx = np.unique(t[::-1], return_index=True)
        keep_idx = (t.size - 1) - np.sort(idx)
    else:
        # indices of the first occurrences
        _, idx = np.unique(t, return_index=True)
        keep_idx = np.sort(idx)
    return ds.isel(time=keep_idx)


def check_group_var(by_group: dict, requests: dict) -> dict:
    """
    Verify that loaded datasets match the user or tracker variable requests.

    Supports both flat and nested ``by_group`` structures:

    - Flat: ``{group -> xr.Dataset}``, ``requests: {group -> spec}``
    - Nested: ``{method -> {group -> xr.Dataset}}``,
      ``requests: {method -> {group -> spec}}`` or ``{group -> spec}``
      (applied to all methods).

    Logs INFO for present variables and a WARNING for missing ones.

    Parameters
    ----------
    by_group : dict
        Loaded datasets, either flat ``{group: xr.Dataset}`` or nested
        ``{method: {group: xr.Dataset}}``.
    requests : dict
        Variable requests in the same shape as ``by_group``, or a flat group
        spec applied to every method when ``by_group`` is nested.

    Returns
    -------
    dict
        Summary with keys:

        - ``"ignored_groups"``: groups present in requests but absent from data.
        - ``"missing_by_group"``: ``{group: [missing_var, ...]}`` for each
          group where at least one requested variable was not loaded.
    """

    def _is_group_spec_map(d: dict) -> bool:
        if not isinstance(d, dict) or not d:
            return False
        return all(
            isinstance(v, dict)
            and any(
                k in v
                for k in ("variables", "level_selection", "level_indices", "level_values", "levels")
            )
            for v in d.values()
        )

    def _is_method_group_spec_map(d: dict) -> bool:
        if not isinstance(d, dict) or not d:
            return False
        return all(_is_group_spec_map(v) for v in d.values())

    def _expected_vars(spec: dict) -> set:
        return set((spec or {}).get("variables") or [])

    if not isinstance(by_group, dict) or not by_group:
        return {"\t\t\tignored_groups": [], "missing_by_group": {}}

    # Detect flat vs nested by_group
    is_flat = all(hasattr(v, "dims") for v in by_group.values() if v is not None)

    if is_flat:
        # Log dims
        for g, ds in by_group.items():
            dims = ", ".join(f"{k}={v}" for k, v in ds.sizes.items())
            logger.info("\t\t\tDataset[%s] : dimensions %s", g, dims)

        # Normalise requests -> {group -> spec}
        req_map = requests or {}
        if not _is_group_spec_map(req_map):
            req_map = {}

        requested_groups = set(req_map.keys())
        loaded_groups = set(by_group.keys())

        ignored = requested_groups - loaded_groups
        if ignored:
            logger.info("\t\t\tGroups ignored (absent from data): %s", ", ".join(sorted(ignored)))

        missing_by_group: dict[str, list[str]] = {}
        for grp in sorted(loaded_groups & requested_groups):
            exp = _expected_vars(req_map.get(grp))
            if not exp:
                continue
            present = set(by_group[grp].data_vars)
            missing = sorted(exp - present)
            if missing:
                logger.warning("\t\t\t[%s] Missing variables: %s", grp, ", ".join(missing))
                missing_by_group[grp] = missing
            else:
                logger.info("\t\t\t[%s] All requested variables are present (%d).", grp, len(exp))

        return {"\t\t\tignored_groups": sorted(ignored), "missing_by_group": missing_by_group}

    # Nested: {method -> {group -> xr.Dataset}}
    # Log dims
    for m, grp_map in by_group.items():
        if not isinstance(grp_map, dict):
            continue
        for g, ds in grp_map.items():
            if not hasattr(ds, "dims"):
                continue
            dims = ", ".join(f"{k}={v}" for k, v in ds.sizes.items())
            logger.info("\t\t\tDataset[%s/%s] : dimensions %s", m, g, dims)

    # Normalise requests
    req_in = requests or {}
    if _is_method_group_spec_map(req_in):
        req_methods = req_in  # {method -> {group -> spec}}
    elif _is_group_spec_map(req_in):
        # Apply the same group block to all present methods
        req_methods = {m: req_in for m in by_group.keys() if isinstance(by_group.get(m), dict)}
    else:
        req_methods = {}

    methods = set(k for k, v in by_group.items() if isinstance(v, dict)) | set(req_methods.keys())

    ignored_flat: list[str] = []
    missing_flat: dict[str, list[str]] = {}

    for m in sorted(methods):
        grp_map = by_group.get(m) if isinstance(by_group.get(m), dict) else {}
        req_map = req_methods.get(m, {})

        requested_groups = set(req_map.keys())
        loaded_groups = set(grp_map.keys())

        ignored = requested_groups - loaded_groups
        if ignored:
            logger.info(
                "\t\t\tGroups ignored (absent from data) for %s: %s",
                m,
                ", ".join(sorted(ignored)),
            )
            ignored_flat.extend([f"{m}/{g}" for g in sorted(ignored)])

        for g in sorted(loaded_groups & requested_groups):
            exp = _expected_vars(req_map.get(g))
            if not exp:
                continue
            present = set(grp_map[g].data_vars)
            missing = sorted(exp - present)
            key = f"{m}/{g}"
            if missing:
                logger.warning("\t\t\t[%s] Missing variables: %s", key, ", ".join(missing))
                missing_flat[key] = missing
            else:
                logger.info("\t\t\t[%s] All requested variables are present (%d).", key, len(exp))

    return {"\t\t\tignored_groups": ignored_flat, "missing_by_group": missing_flat}
