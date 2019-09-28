###########
ptm-viewer
###########

Lightweight PTM viewer based on Pyside2 and OpenGL

General view of the interface
==============================

Lightweight ptm viewer based on PySide2 and OpenGL

General view of the interface

Viewing ptm under different light conditions


Installation
=============

See `Installation <docs/install.rst>`_


Quick Start
===========

- Follow the installation procedure detailed in `here <docs/install.rst>`_
- Activate your conda environment from terminal
- Go to repository
- `cd ptmviewer`
- `python qtapp.py`

Roadmap
========

- Handle more advanced shading.

  - PBR if possible

- Change light controller

  - Give more control on illumination parameters


Known Issues
=============

- Camera pierces through the ptm surface
- Light also passes through ptm surface.


Remarks
========

The rendering should work in real time with no hassle since its all done in
OpenGL. Just make sure that you have OpenGL 3.3 and above. Your driver
information is also shown in status bar during the loading of a ptm.
