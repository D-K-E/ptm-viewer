# author: Kaan Eraslan

# Purpose: Application wrapper for ptm viewer

# glwidget class

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtGui import QOpenGLWidget

# Opengl drawing related
from PySide2.QtGui import QVector3D
from PySide2.QtGui import QOpenGLVertexArrayObject
from PySide2.QtGui import QOpenGLBuffer
from PySide2.QtGui import QOpenGLShaderProgram
from PySide2.QtGui import QOpenGLShader
from PySide2.QtGui import QOpenGLContext
from PySide2.QtGui import QMatrix4x4
from PySide2.QtGui import QVector4D
from PySide2.QtGui import QColor


from PySide2.QtWidgets import QApplication
from PySide2.QtWidgets import QMessageBox
from PySide2.QtWidgets import QOpenGLWidget

from PySide2.QtCore import QCoreApplication


from shiboken2 import VoidPtr
import sys
import numpy as np
import os

import ctypes


try:
    from OpenGL import GL as pygl
except ImportError:
    app = QtWidgets.QApplication(sys.argv)
    messageBox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "OpenGL hellogl",
                                       "PyOpenGL must be installed to run this example.",
                                       QtWidgets.QMessageBox.Close)
    messageBox.setDetailedText(
        "Run:\npip install PyOpenGL PyOpenGL_accelerate")
    messageBox.exec_()
    sys.exit(1)


class GLWidget(QOpenGLWidget):
    "qopengl widget"

    def __init__(self, parent=None):
        QOpenGLWidget.__init__(parent)
        pass
