#################
Installation
#################

Quick Start
============

For those using :code:`conda` package as package manager
the installation is quite easy:

- :code:`git clone https://github.com/D-K-E/ptm-viewer.git`

- :code:`cd PATH_TO_REPO/ptm-viewer`

- :code:`conda create --name ptmviewer --file ptmviewer-spec-file.txt`

- :code:`conda activate ptmviewer`

- :code:`cd ptmviewer`

- :code:`python qtapp.py`


Requirements for Existing Environments
=======================================

If you want to conserve your working environment, the spec file contains all
the libraries necessary to run the interface.

You can install the libraries by:

- :code:`conda install --name YOUR_ENV_NAME --file ptmviewer-spec-file.txt`
