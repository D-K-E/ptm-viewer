# ptm-viewer
Lightweight ptm viewer based on numpy and tkinter

General view of the interface

![Interface Image](docs/ptmviewer-1.png?raw=true "Interface image")


Viewing ptm under different light conditions

![Light Image](docs/ptmviewer-2.png?raw=true "Light image")

![Light Image 2](docs/ptmviewer-3.png?raw=true "Light image 2")

## Installation

See ![Docs](docs/install.rst "Installation")


## Quick Start

- Follow the installation proceedure detailed in ![here](docs/install.rst "installation")
- Activate your conda environment from terminal
- Go to repository
- `cd main`
- `python app.py`

## Remarks

The rendering is extremely slow due to numpy implementation of the backend.
There are plans to use opengl as backend in the future. 

For developpers that want to use the numpy backend to manipulate their own
images, please see the `main/core.py`. The interface is simply a GUI wrapper
around some of the functionalities covered there.
