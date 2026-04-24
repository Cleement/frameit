import logging
from pathlib import Path

import pytest

CONFIG_DIR = Path("configs_ktests")


def pytest_addoption(parser):
    parser.addoption(
        "--family",
        action="store",
        default="all",
        choices=["MNH", "AROME", "all"],
        help="Famille de configs à tester",
    )


def _list_configs():
    if not CONFIG_DIR.exists():
        return []
    return sorted(str(p) for p in CONFIG_DIR.glob("*.yml"))


def _split_families(paths):
    mnh = [p for p in paths if "mnh" in Path(p).stem.lower()]
    arome = [p for p in paths if "arome" in Path(p).stem.lower()]
    return mnh, arome


def _select_configs_from_config(config):
    all_configs = _list_configs()
    mnh, arome = _split_families(all_configs)
    family = config.getoption("--family")
    if family == "MNH":
        return mnh
    if family == "AROME":
        return arome
    return all_configs  # all


def pytest_generate_tests(metafunc):
    if "conf_path" in metafunc.fixturenames:
        params = _select_configs_from_config(metafunc.config)
        ids = [Path(p).stem for p in params]
        metafunc.parametrize("conf_path", params, ids=ids)


@pytest.fixture(scope="session")
def artifact_logger():
    """Logger unique pour tous les tests, écrit en append dans tests/_artifacts/frameit.log"""
    artifacts_root = Path("tests") / "_artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    log_path = artifacts_root / "frameit.log"

    if log_path.exists():
        log_path.unlink()

    logger = logging.getLogger("FrameIt.artifacts")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fh_pref = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh_pref.setFormatter(logging.Formatter("[FrameIt] %(message)s"))

    fh_nop = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh_nop.setFormatter(logging.Formatter("%(message)s"))

    class OnlyWithPrefix(logging.Filter):
        def filter(self, record):
            return not getattr(record, "noprefix", False)

    class OnlyNoPrefix(logging.Filter):
        def filter(self, record):
            return getattr(record, "noprefix", False)

    fh_pref.addFilter(OnlyWithPrefix())
    fh_nop.addFilter(OnlyNoPrefix())

    logger.handlers = [fh_pref, fh_nop]

    try:
        yield logger, log_path
    finally:
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)
