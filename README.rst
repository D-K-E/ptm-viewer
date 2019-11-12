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
- From the location of `setup.py` at the main folder, start the program with 
  `python ptmviewer/qtapp.py`

Roadmap
========

Milestone 0.1.0
-------------

- [X] Smoother UI with dock widgets

- [] Reproducible lightening conditions using lightning parameter serialization

- [] Handle light and camera rotation in all axes.

- [] Handle light and camera movement in all axes.

- [X] Small notepad for taking notes about the object as it is seen by the user.



Milestone 0.2.0
---------------

- Add a couple of other shaders.

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
