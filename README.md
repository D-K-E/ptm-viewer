# ptm-viewer

Lightweight ptm viewer based on Pyside2 and OpenGL

General view of the interface
=======

Lightweight ptm viewer based on PySide2 and OpenGL

General view of the interface

Viewing ptm under different light conditions


## Installation

See ![Docs](docs/install.rst "Installation") for installation.


## Quick Start

- Follow the installation procedure detailed in ![here](docs/install.rst "installation")
- Activate your conda environment from terminal
- Go to repository
- `cd ptmviewer`
- `python qtapp.py`

## Remarks

The rendering should work with no hassle since its all done in OpenGL. Just
make sure that you have OpenGL 3.3 and above. Your driver information is also
shown in status bar during the loading of a ptm.

Only RGB ptm's are supported.

## Known Issues

- Shader selection does not work.
- Rendering gets weird when fragment values pass above 1.0
- Camera and light source can pass through texture board.
