FrameIt Quick Start
===================

This page describes the minimal workflow to run **FrameIt** using its command-line interface (CLI).

If FrameIt is not installed yet, see :doc:`installation` first.

Prerequisites
-------------

- A working Python environment where FrameIt is installed.
- A FrameIt configuration file in YAML format (referred to as ``config.yml`` below).

Check that the CLI is available
-------------------------------

After activating your environment, verify that the ``frameit`` command is visible:

.. code-block:: bash

   frameit -h

If you cannot access the ``frameit`` executable (PATH issues on some HPC systems), you can use:

.. code-block:: bash

   python -m frameit -h

Display environment information
-------------------------------

To print the FrameIt version and key dependency versions:

.. code-block:: bash

   frameit info

Prepare the YAML configuration
------------------------------

1. Start from an example YAML shipped with the repository (see ``conf_example.yml``), or copy an existing configuration.
2. Update the following elements according to your case:

   - Input data locations (paths or patterns).
   - Output directory (``frameit_output_dir``).
   - Requested variables and processing options (tracking, extraction, polar projection, etc.).

Validate the configuration
--------------------------

FrameIt provides a lightweight configuration check that loads the YAML, checks right of writting in the output directory and applies the model preset:

.. code-block:: bash

   frameit validate path/to/config.yml

If the configuration is valid, the command prints:

.. code-block:: text

   OK: configuration loaded successfully

Run FrameIt
-----------

Run the pipeline with:

.. code-block:: bash

   frameit run path/to/config.yml

Logging is initialized automatically and written under the output directory specified in the configuration.
At the end of the run, FrameIt prints a runtime summary (timer report) in the logs.

NetCDF exports
--------------

By default, the CLI runs NetCDF exports at the end of the pipeline:

- NetCDF export: enabled by default
- Polar export: enabled by default
- Cartesian export: enabled by default

To disable some exports:

.. code-block:: bash

   # Disable all NetCDF exports
   frameit run path/to/config.yml --no-export-netcdf

.. code-block:: bash

   # Disable only polar exports
   frameit run path/to/config.yml --no-export-polar

.. code-block:: bash

   # Disable only cartesian exports
   frameit run path/to/config.yml --no-export-cart

You may also tune compression:

.. code-block:: bash

   frameit run path/to/config.yml --compress-level N

Where N is the level of compression you want. The default is set to N=1.
Note that increasing the level of compression could drastically increase the execution time of FrameIt.

Optional: HDF5 debug environment variable
-----------------------------------------

The CLI removes the ``HDF5_DEBUG`` environment variable by default to avoid excessively verbose HDF5 logging.
If you want to keep it unchanged:

.. code-block:: bash

   frameit run path/to/config.yml --no-hdf5-debug-pop

Getting help for subcommands
----------------------------

For command help and available options:

.. code-block:: bash

   frameit -h
   frameit run -h
   frameit validate -h
   frameit info -h

Exit codes
----------

- ``0``: success.
- Non-zero: an error occurred (details are reported in the FrameIt logs and/or printed to stderr).

Minimal HPC example (Slurm)
---------------------------

A typical Slurm wrapper may look like:

.. code-block:: bash

   #!/bin/bash
   #SBATCH -J frameit
   #SBATCH --time=02:00:00
   #SBATCH --nodes=1
   #SBATCH --ntasks=1

   source /path/to/conda.sh
   conda activate frameit_env

   frameit validate /path/to/config.yml
   frameit run /path/to/config.yml

This example is intentionally minimal. Adapt resources and modules according to your HPC environment and workload.

