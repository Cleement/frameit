Installation of FrameIt
=======================

In this section, we describe the minimal steps required to install **FrameIt** for development and reproducible scientific workflows.

1. Prerequisites
----------------

- **Python**: 3.10 or newer.
- **Git**: to clone the repository.
- **a dedicated environment manager** : conda/mamba.

2. Get the source code
----------------------

In a terminal, go to your installation directory and run:

.. code-block:: bash

   git clone <FrameIt_REPO_URL>
   cd frameit

You should end up with a structure similar to:

.. code-block:: text

   .
   ├── docs
   ├── environment.yaml
   ├── pyproject.toml
   ├── scripts
   └── src

The file ``environment.yaml`` define the dependencies required by **FrameIt**.

3. Create and activate the Python environment using a conda/mamba environment
-----------------------------------------------------------------------------


If you already have conda/mamba installed, use the file ``environment.yaml``:

.. code-block:: bash

   conda env create -f environment.yaml
   conda activate frameit_env


4. Install the FrameIt package
------------------------------

From the repository root (where ``pyproject.toml`` is located):

.. code-block:: bash

   pip install -e .

Quick verification:

.. code-block:: bash

   python -c "import frameit; print('frameit import OK')"
