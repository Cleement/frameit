from pathlib import Path

from frameit.core.settings_class import SimulationConfig


def test_build_pattern_contains_wildcards():
    conf = SimulationConfig.from_yaml(Path("configs_ktests/arome_oper_batsirai.yml"))
    pattern = conf.build_pattern()
    assert "???" in pattern or "*" in pattern
