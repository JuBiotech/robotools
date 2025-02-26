Welcome to the ``robotools`` documentation!
===========================================

.. image:: https://img.shields.io/pypi/v/robotools
   :target: https://pypi.org/project/robotools

.. image:: https://img.shields.io/badge/code%20on-Github-lightgrey
   :target: https://github.com/JuBiotech/robotools

.. image:: https://zenodo.org/badge/358629210.svg
   :target: https://zenodo.org/badge/latestdoi/358629210


``robotools`` is a Python package for planning *in silico* liquid handling operations.

It can create Tecan Freedom EVO or Tecan Fluent worklist files on the fly, making it possible to program complicated liquid handling operations
from Python scripts or Jupyter notebooks.

Installation
============

.. code-block:: bash

   pip install robotools

You can also download the latest version from `GitHub <https://github.com/JuBiotech/robotools>`_.

Tutorials
=========

In the following chapters, we introduce the data structures, worklist-creation and extra features.

.. toctree::
   :maxdepth: 1

   notebooks/01_Labware_Basics
   notebooks/02_Worklist_Basics
   notebooks/03_Large_Volumes
   notebooks/04_Composition_Tracking
   notebooks/05_DilutionPlan
   notebooks/06_Advanced_Worklist_Commands
   notebooks/07_TipMasks


API Reference
=============

.. toctree::
   :maxdepth: 2

   robotools_evotools
   robotools_fluenttools
   robotools_liquidhandling
   robotools_worklists
   robotools_transform
   robotools_utils
