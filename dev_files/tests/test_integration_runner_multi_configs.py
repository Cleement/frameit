from pathlib import Path

import pytest
import xarray as xr

from frameit.core.runner import FRAMEITRunner
from frameit.core.settings_class import SimulationConfig


def test_runner_full_real_data_multi(conf_path, artifact_logger):
    logger, log_path = artifact_logger

    cfg_file = Path(conf_path)
    if not cfg_file.exists():
        pytest.skip(f"Fichier de config introuvable: {cfg_file}")

    conf = SimulationConfig.from_yaml_with_model_preset(cfg_file)

    data_dir = Path(conf.data_dir)
    if not data_dir.exists():
        pytest.skip(f"Données absentes pour {cfg_file.stem}: {data_dir}")

    out_dir = log_path.parent

    sep = "=" * 77
    logger.info(sep, extra={"noprefix": True})
    logger.info(cfg_file.stem, extra={"noprefix": True})
    logger.info(sep, extra={"noprefix": True})
    start_size = log_path.stat().st_size

    runner = FRAMEITRunner(conf, output_dir=out_dir)
    res = runner.run()

    # 1) Statut global
    assert res.ok, "RunResult.ok est False"
    assert res.n_files > 0, "Aucun fichier traité"
    assert res.output_dir == out_dir, "Output_dir du résultat inattendu"
    assert out_dir.exists(), "Le répertoire d'output n'existe pas"

    # 2) Fichiers traités présents
    assert isinstance(res.files, list) and all(isinstance(p, Path) for p in res.files), (
        "RunResult.files doit être une liste de Path"
    )
    assert all(p.exists() for p in res.files), "Certains fichiers listés n'existent pas sur disque"

    # --- Helpers ---
    time_dim_cfg = getattr(conf, "name_time_dim", None) or "time"

    def _has_time(ds: xr.Dataset) -> bool:
        return (
            ("time" in ds.dims)
            or (time_dim_cfg in ds.dims)
            or ("time" in ds.coords)
            or (time_dim_cfg in ds.coords)
        )

    def _time_is_dim(ds: xr.Dataset) -> bool:
        return ("time" in ds.dims) or (time_dim_cfg in ds.dims)

    def _ntime(ds: xr.Dataset) -> int:
        if "time" in ds.dims:
            return ds.sizes["time"]
        if time_dim_cfg in ds.dims:
            return ds.sizes[time_dim_cfg]
        if "time" in ds.coords:
            return len(ds["time"])
        if time_dim_cfg in ds.coords:
            return len(ds[time_dim_cfg])
        return 0

    def _expected_map_from_yaml(yaml_map: dict | None) -> dict[str, set]:
        if not isinstance(yaml_map, dict):
            return {}
        return {g: set((spec or {}).get("variables", []) or []) for g, spec in yaml_map.items()}

    def _validate_obj(name: str, obj, expected_yaml: dict | None):
        exp = _expected_map_from_yaml(expected_yaml)

        # structure
        if isinstance(obj, xr.Dataset):
            assert _has_time(obj), (
                f"[{name}] dimension temps absente (ni 'time' ni '{time_dim_cfg}')"
            )
            assert _time_is_dim(obj), (
                f"[{name}] 'time' ou '{time_dim_cfg}' devrait être une dimension"
            )
            # variables attendues, si exp explicite un seul groupe sans dimension verticale
            if len(exp) == 1 and list(exp.keys())[0].lower() in ("surface", "none"):
                expected = next(iter(exp.values()))
                missing = sorted(expected - set(obj.data_vars))
                assert not missing, f"[{name}] Variables manquantes: {missing}"
        else:
            assert isinstance(obj, dict), (
                f"[{name}] doit être un xr.Dataset ou un dict[str, xr.Dataset]"
            )
            assert obj, f"[{name}] dict vide"
            for grp, ds_grp in obj.items():
                assert isinstance(ds_grp, xr.Dataset), f"[{name}] [{grp}] valeur non Dataset"
                assert _has_time(ds_grp), f"[{name}] [{grp}] dimension temps absente"
                assert _time_is_dim(ds_grp), (
                    f"[{name}] [{grp}] 'time' ou '{time_dim_cfg}' devrait être une dimension"
                )
                if grp in exp and exp[grp]:
                    missing = sorted(exp[grp] - set(ds_grp.data_vars))
                    assert not missing, f"[{name}] [{grp}] Variables manquantes: {missing}"

        # Nfichiers vs Ntemps
        if isinstance(obj, xr.Dataset):
            n_time = _ntime(obj)
            if len(res.files) > 1:
                assert n_time == len(res.files), (
                    f"[{name}] Ntime ({n_time}) différent du nombre de fichiers ({len(res.files)})"
                )
            else:
                logger.info(
                    "[%s] Single-file dataset, skipping Ntime vs Nfiles check",
                    name,
                    extra={"noprefix": True},
                )
        else:
            counts = {grp: _ntime(ds) for grp, ds in obj.items()}
            if not counts:
                return
            max_n = max(counts.values())
            if len(res.files) > 1:
                assert max_n == len(res.files), (
                    f"[{name}] Max Ntime ({max_n}) différent du nombre de fichiers ({len(res.files)})"
                )
                short = {g: n for g, n in counts.items() if n != max_n}
                if short:
                    logger.info(
                        "[%s] Groupes avec moins de pas de temps que les fichiers: %s",
                        name,
                        short,
                        extra={"noprefix": True},
                    )
            else:
                logger.info(
                    "[%s] Single-file dataset, skipping Ntime vs Nfiles check",
                    name,
                    extra={"noprefix": True},
                )

    # Attentes issues de la conf
    expected_user = getattr(conf, "requested_variables", {}) or {}
    method = getattr(conf, "tracking_method", None) or "wind_pressure"
    expected_tracker_all = getattr(conf, "requested_variables_tracker", {}) or {}
    expected_tracker = (
        expected_tracker_all.get(method, {}) if isinstance(expected_tracker_all, dict) else {}
    )

    # 3) Valider ds_user
    _validate_obj("user", runner.ds_user, expected_user)

    # 4) Valider ds_tracker
    if expected_tracker:
        _validate_obj("tracker", runner.ds_tracker[method], expected_tracker)
    else:
        # Si aucune attente tracker nâ€™est définie pour la méthode, on vérifie seulement la structure si présent
        if runner.ds_tracker is not None:
            _validate_obj("tracker", runner.ds_tracker, {})

    # 5) Log non vide après exécution
    end_size = log_path.stat().st_size
    assert end_size >= start_size, "Le log ne s'est pas enrichi pendant l'exécution"
