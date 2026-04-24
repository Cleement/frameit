# #!/usr/bin/env python3
"""
Created on Fri mai 24 2024
@author: Clément soufflet

"""

import logging
import os
from pathlib import Path

from frameit.core.runner import FrameitRunner
from frameit.core.settings_class import SimulationConfig
from frameit.io.netcdf_export import export_outputs
from frameit.utils import setup_FRAMEIT_logging

os.environ.pop("HDF5_DEBUG", None)


def main():
    conf = SimulationConfig.from_yaml_with_model_preset(Path("../configs_ktests/mnh_ianos.yml"))

    # AROME
    conf = SimulationConfig.from_yaml_with_model_preset(Path("../configs_ktests/arome_oper_batsirai.yml"))
    # conf = SimulationConfig.from_yaml_with_model_preset(Path("../configs_ktests/arome_oper_belna.yml"))
    # MNH
    # conf = SimulationConfig.from_yaml_with_model_preset(Path("../configs_ktests/mnh_chido.yml"))
    # conf = SimulationConfig.from_yaml_with_model_preset(Path("../configs_ktests/mnh_ianos.yml"))

    out_dir = Path(conf.frameit_output_dir).resolve()

    log_path = setup_FRAMEIT_logging(
        conf.frameit_output_dir,
        level=("DEBUG" if conf.DEBUG else "INFO"),
        simu_name=getattr(conf, "simulation_name", None),
    )
    logging.getLogger("frameit").info("Logging initialized: %s", str(log_path))

    runner = FrameitRunner(conf)
    runner.run()
    with runner.timer.section("Netcdf export"):
        export_outputs(
            runner,
            institution="LACy, Université de La Réunion, CNRS, Météo-France",
            out_dir=out_dir,
            export_polar=True,
            export_cart=True,
            compress_level=1,
        )

    logging.getLogger("frameit").info("\n\nNetCDF export done.\n\n")

    runner.timer.log_summary(logging.getLogger("frameit"), title="FrameIt runtime summary")

    logging.getLogger("frameit").info("\n\nFrameIt ends correctly\n\n")


if __name__ == "__main__":
    main()
