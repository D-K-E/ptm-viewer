# author: Kaan Eraslan

# Purpose: Application wrapper for ptm viewer

from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL

from PySide2.QtGui import QOpenGLBuffer
from PySide2.QtGui import QOpenGLContext
from PySide2.QtGui import QOpenGLShader
from PySide2.QtGui import QOpenGLShaderProgram
from PySide2.QtGui import QOpenGLVertexArrayObject
from PySide2.QtGui import QOpenGLWidget

from PySide2.QtCore import QCoreApplication

from shiboken2 import VoidPtr

import numpy as np
import sys
import os

try:
    from OpenGL import GL as gl
except ImportError:
    app = QtWidgets.QApplication(sys.argv)
    messageBox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "OpenGL hellogl",
                                       "PyOpenGL must be installed to run this example.",
                                       QtWidgets.QMessageBox.Close)
    messageBox.setDetailedText("Run:\npip install PyOpenGL PyOpenGL_accelerate")
    messageBox.exec_()
    sys.exit(1)

from interface.brutInterface import Ui_MainWindow


class AppWindowInit(Ui_MainWindow):
    """
    Initializes the image window
    made in qt designer
    """

    def __init__(self):
        self.main_window = QtCore.Qt.QMainWindow()
        super().setupUi(self.main_window)
        pass


class GLWidget(QOpenGLWidget):
    "Extends qopengl widget to take light source etc as arguments"
    lightXPosChanged = QtCore.Signal(int)
    lightYPosChanged = QtCore.Signal(int)
    lightZPosChanged = QtCore.Signal(int)
    lightIntensityChanged = QtCore.Signal(float)
    ambientTermChanged = QtCore.Signal(float)
    diffuseCoeffChanged = QtCore.Signal(float)
    shininessChanged = QtCore.Signal(float)
    zoomChanged = QtCore.Signal(float)
    specColorChanged = QtCore.Signal(float)

    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
        self.object = 0
        self.xpos = 0
        self.ypos = 0
        self.zpos = 0
        self.diffuse_coeff = 0.08
        self.specColor = 0.1
        self.ambientTerm = 0.1
        self.lastPos = QtCore.QPoint()
        self.context = QtGui.QOpenGLContext(self)
        self.vao = QtGui.QOpenGLVertexArrayObject(self)
        self.vbo = QtGui.QOpenGLBuffer(QtGui.QOpenGLBuffer.VertexBuffer)
        self.ebo = QtGui.QOpenGLBuffer(QtGui.QOpenGLBuffer.IndexBuffer)
        self.program = None
        self.texture = None
        self.vertices = [
            # vertex Pos xyz || normal xyz || texCoord xy
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        ]
        self.indices = [0, 1, 3, 1, 2, 3]

        self.funcs = None
        self.coords = ((-1, 1), (-1, -1), (1, -1), (1, 1))
        self.texCoords = ((0, 1), (0, 0), (1, 0), (1, 1))
        self.clearColor = QtGui.QColor()
        projectdir = os.getcwd()
        viewerdir = os.path.join(projectdir, 'ptmviewer')
        assetsdir = os.path.join(viewerdir, "assets")
        self.shadersdir = os.path.join(assetsdir, 'shaders')
        self.shaders = {
            "blinnPhong": {
                "vertex": os.path.join(self.shadersdir, "blinnPhong.vert"),
                "fragment": os.path.join(self.shadersdir, "blinnPhong.frag")
            },
            "phong": {
                "vertex": os.path.join(self.shadersdir, "phong.vert"),
                "fragment": os.path.join(self.shadersdir, "phong.frag")
            },
            "lamp": {
                "vertex": os.path.join(self.shadersdir, "lamp.vert"),
                "fragment": os.path.join(self.shadersdir, "lamp.frag")
            }
        }

    def getGlInfo(self):
        "Get opengl info"
        info = """
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
            """.format(
            gl.glGetString(gl.GL_VENDOR),
            gl.glGetString(gl.GL_RENDERER),
            gl.glGetString(gl.GL_VERSION),
            gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
        )
        return info

    def loadTexture(self, texturePath, textureType):
        "Load texture path"
        qimage = QtGui.QImage(texturePath)
        texture = QtGui.QOpenGLTexture(qimage.mirrored())
        texture.setMinificationFilter(
            QtGui.QOpenGLTexture.NearestMipMapLinear)
        texture.setMinificationFilter(
            QtGui.QOpenGLTexture.NearestMipMapNearest)
        return texture

    def loadVertexShader(self, shaderName: str):
        "Load the vertex shader from available shaders"
        vshaderPath = self.shaders[shaderName]['vertex']
        vshader = QtGui.QOpenGLShader(QtGui.QOpenGLShader.Vertex)
        isCompiled = vshader.compileSourceFile(vshaderPath)
        if isCompiled is False:
            raise ValueError(
                "Vertex shader {0} in {1} is not compiled".format(
                    shaderName, vshaderPath)
            )
        return vshader

    def loadFragmentShader(self, shaderName: str):
        "Load fragment shader"
        fragPath = self.shaders[shaderName]['fragment']
        fshader = QtGui.QOpenGLShader(QtGui.QOpenGLShader.Fragment)
        isCompiled = fshader.compileSourceFile(fragPath)
        if isCompiled is False:
            raise ValueError(
                "Fragment shader {0} in {1} is not compiled".format(
                    shaderName, fragPath)
            )
        return fshader

    def loadShader(self, shaderName: str, shaderType: str):
        "Load shader either fragment or vertex"
        stype = shaderType.lower()
        if (stype is not "vertex" and stype is not "fragment"):
            raise ValueError(
                "Shader type {0} is unknown, provide either "
                "vertex or fragment as value".format(shaderType)
            )
        if stype == "fragment":
            return self.loadFragmentShader(shaderName)
        else:
            return self.loadVertexShader(shaderName)

    def resizeGL(self, width, height):
        "resize gl widget"
        side = min(width, height)
        if side < 0:
            return

        gl.glViewport((width - side) // 2, (height - side) // 2, side,
                      side)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-0.5, +0.5, +0.5, -0.5, 4.0, 15.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.setXPos(self.xpos + 8 * dy)
            self.setYPos(self.ypos + 8 * dx)
        elif event.buttons() & Qt.RightButton:
            self.setXPos(self.xpos + 8 * dy)
            self.setZPos(self.zpos + 8 * dx)

        self.lastPos = event.pos()

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

    def setXPos(self, angle):
        "Set x angle position"
        normed = self.normalizeAngle(angle)
        if angle != self.xpos:
            self.xpos = angle
            self.lightXPosChanged.emit(angle)
            self.update()

    def setYPos(self, angle):
        "Set y angle position"
        normed = self.normalizeAngle(angle)
        if angle != self.ypos:
            self.ypos = angle
            self.lightYPosChanged.emit(angle)
            self.update()

    def setZPos(self, angle):
        "Set z angle position"
        normed = self.normalizeAngle(angle)
        if angle != self.zpos:
            self.zpos = angle
            self.lightZPosChanged.emit(angle)
            self.update()

    def setSpecColor(self, val: float):
        "Set spec color"
        if val != self.specColor:
            self.specColor = val
            self.specColorChanged.emit(val)
            self.update()

    def setDiffuseCoeff(self, val: float):
        "Set diffuse coeff color"
        if val != self.diffuse_coeff:
            self.diffuse_coeff = val
            self.diffuseCoeffChanged.emit(val)
            self.update()

    def setShininess(self, val: float):
        "Set diffuse coeff color"
        if val != self.shininess:
            self.shininess = val
            self.shininessChanged.emit(val)
            self.update()

    def setAmbientTerm(self, val: float):
        "Set ambient term color"
        if val != self.ambientTerm:
            self.ambientTerm = val
            self.ambientTermChanged.emit(val)
            self.update()

    def paintGL(self):
        pass

    def initializeGL(self):
        self.context.makeCurrent()
        self.funcs = QtGui.QOpenGLFunctions(self.context)
        self.funcs.initializeOpenGLFunctions()
        
        # create vao, vbo, ebo
        isVao = self.vao.create()
        if isVao is False:
            raise ValueError("vertex array object is not created")
        isVbo = self.vbo.create()
        if isVbo is False:
            raise ValueError("vertex buffer object is not created")
        isEbo = self.ebo.create()
        if isEbo is False:
            raise ValueError("element buffer object is not created")
        # bind object
        self.vao.bind()
        self.vbo.bind()
        self.ebo.bind()

        # read data
        self.vbo.read(0, self.vertices, len(self.vertices))
        self.ebo.read(0, self.indices, len(self.indices))

        # set usage pattern: memory type
        self.vbo.setUsagePattern(QtGui.QOpenGLBuffer.StaticDraw)
        self.ebo.setUsagePattern(QtGui.QOpenGLBuffer.StaticDraw)

        # setting program to context
        self.program = QtGui.QOpenGLShaderProgram(self.context)

        # let's specify attributes

        # aPos in vertex shader
        self.funcs.glVertexAttribPointer(0, 3,
                                         self.funcs.GL_FLOAT,
                                         self.funcs.GL_FALSE,
                                         8, 0)
        self.funcs.glEnableVertexAttribArray(0)

        # normals in vertex shader
        self.funcs.glVertexAttribPointer(1, 3,
                                         self.funcs.GL_FLOAT,
                                         self.funcs.GL_FALSE,
                                         8, 3)
        self.funcs.glEnableVertexAttribArray(1)

    def makeObject(self):
        pass


class AppWindowFinal(AppWindowInit):
    "Final window"

    def __init__(self):
        super().__init__()
        self.ptmfiles = {}
        self.ptm = None
        self.image = None
        self.pilimg = None

        # Main Window Events
        self.main_window.setWindowTitle("Python PTM Viewer")
        self.main_window.closeEvent = self.closeApp

        # Button related events
        self.importBtn.clicked.connect(self.browseFolder)
        self.importBtn.setShortcut('ctrl+o')

    ### Standard Gui Elements ###

    def showInterface(self):
        "Show the interface"
        self.main_window.show()

    def browseFolder(self):
        "Import ptm files from folder using file dialog"
        self.listWidget.clear()
        fdir = QtWidgets.QFileDialog.getOpenFileNames(
            self.centralwidget,
            "Select PTM files", "", "PTMs (*.ptm)")
        if fdir:
            for fname in fdir[0]:
                ptmitem = QtWidgets.QListWidgetItem(self.listWidget)
                itemname = os.path.basename(fname)
                ptmitem.setText(itemname)
                ptmobj = {}
                ptmobj['path'] = fname
                ptmobj['name'] = itemname
                ptmobj['index'] = self.listWidget.indexFromItem(ptmitem)
                self.ptmfiles[ptmobj['index']] = ptmobj
                self.listWidget.sortItems()

    def closeApp(self, event):
        "Close application"
        reply = QtWidgets.QMessageBox.question(self.centralwidget, 'Message',
                                               "Are you sure to quit?", QtWidgets.QMessageBox.Yes |
                                               QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
            sys.exit(0)
        else:
            event.ignore()
            #
        return


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AppWindowFinal()
    window.showInterface()
    sys.exit(app.exec_())
