# author: Kaan Eraslan

# Purpose: Application wrapper for ptm viewer

# glwidget class


# Opengl drawing related
from PySide2.QtGui import QVector3D
from PySide2.QtGui import QVector2D
from PySide2.QtGui import QOpenGLVertexArrayObject
from PySide2.QtGui import QOpenGLBuffer
from PySide2.QtGui import QOpenGLShaderProgram
from PySide2.QtGui import QOpenGLShader
from PySide2.QtGui import QOpenGLTexture
from PySide2.QtGui import QOpenGLContext
from PySide2.QtGui import QMatrix4x4
from PySide2.QtGui import QVector4D
from PySide2.QtGui import QColor
from PySide2.QtGui import QImage


from PySide2.QtWidgets import QApplication
from PySide2.QtWidgets import QMessageBox
from PySide2.QtWidgets import QOpenGLWidget

from PySide2.QtCore import QCoreApplication
from PySide2 import QtCore


from shiboken2 import VoidPtr
import sys
import numpy as np
import os

import ctypes
from ptmviewer.utils.camera import QtCamera
from ptmviewer.utils.light import QtLightSource
from ptmviewer.utils.shaders import shaders

try:
    from OpenGL import GL as pygl
except ImportError:
    app = QtWidgets.QApplication(sys.argv)
    messageBox = QtWidgets.QMessageBox(
        QtWidgets.QMessageBox.Critical,
        "OpenGL hellogl",
        "PyOpenGL must be installed to run this example.",
        QtWidgets.QMessageBox.Close,
    )
    messageBox.setDetailedText(
        "Run:\npip install PyOpenGL PyOpenGL_accelerate"
    )
    messageBox.exec_()
    sys.exit(1)

from PySide2.QtWidgets import QOpenGLWidget


class PtmGLWidget(QOpenGLWidget):
    "OpenGL widget"

    def __init__(self, 
            vertices: np.ndarray,
            indices: np.ndarray,
            parent=None):
        QOpenGLWidget.__init__(self, parent)
        # camera
        self.camera = QtCamera()
        self.camera.position = QVector3D(0.0, 0.0, 3.0)
        self.camera.front = QVector3D(0.0, 0.0, -1.0)
        self.camera.up = QVector3D(0.0, 1.0, 0.0)
        self.camera.movementSensitivity = 0.05

        # light source: point light
        self.lamp = QtLightSource()
        self.shininess = 30.0
        self.ambientCoeff = 0.2

        # shaders
        self.shaders = shaders
        self.attrLoc = {
            "aPos": {"stride": 3, "offset": 0, "layout": 0},
            "acoeff1r": {"stride": 1, "offset": 3, "layout": 1},
            "acoeff2r": {"stride": 1, "offset": 4, "layout": 2},
            "acoeff3r": {"stride": 1, "offset": 5, "layout": 3},
            "acoeff4r": {"stride": 1, "offset": 6, "layout": 4},
            "acoeff5r": {"stride": 1, "offset": 7, "layout": 5},
            "acoeff6r": {"stride": 1, "offset": 8, "layout": 6},
            "acoeff1g": {"stride": 1, "offset": 9, "layout": 7},
            "acoeff2g": {"stride": 1, "offset": 10, "layout": 8},
            "acoeff3g": {"stride": 1, "offset": 11, "layout": 9},
            "acoeff4g": {"stride": 1, "offset": 12, "layout": 10},
            "acoeff5g": {"stride": 1, "offset": 13, "layout": 11},
            "acoeff6g": {"stride": 1, "offset": 14, "layout": 12},
            "acoeff1b": {"stride": 1, "offset": 15, "layout": 13},
            "acoeff2b": {"stride": 1, "offset": 16, "layout": 14},
            "acoeff3b": {"stride": 1, "offset": 17, "layout": 15},
            "acoeff4b": {"stride": 1, "offset": 18, "layout": 16},
            "acoeff5b": {"stride": 1, "offset": 19, "layout": 17},
            "acoeff6b": {"stride": 1, "offset": 20, "layout": 18},
        }
        self.rowsize = 0
        for aName, aprop in self.attrLoc.items():
            self.rowsize += aprop["stride"]
        # opengl data

        self.context = QOpenGLContext()
        self.lampVao = QOpenGLVertexArrayObject()
        self.vao = QOpenGLVertexArrayObject()
        self.lampVbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)

        self.program = QOpenGLShaderProgram()
        self.lampProgram = QOpenGLShaderProgram()
        # vertices
        self.vertices = vertices
        self.indices = indices

        self.lampVertices = np.array(
            [  # first square for cube
                -0.5,  # first corner
                -0.5,
                -0.5,
                0.5,  # second corner
                -0.5,
                -0.5,
                0.5,  # third corner
                0.5,
                -0.5,
                0.5,  # fourth corner
                0.5,
                -0.5,
                # second square for cube
                -0.5,
                0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                -0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                0.5,
                -0.5,
                0.5,
                -0.5,
            ],
            dtype=ctypes.c_float,
        )
        self.rotVectorLamp = QVector3D(0.1, 0.2, 0.5)
        self.rotationAngle = 45.0
        #
        self.worldLimitTop = 3.0 + imTopY
        self.worldLimitBottom = -3.0 + imBottomY
        self.worldLimitRight = 3.0 + imRightX
        self.worldLimitLeft = -3.0 + imLeftX
        self.worldLimitDepthNeg = -20.0
        self.worldLimitDepthPos = 20.0

    def loadShader(self, shaderName: str, shaderType: str, fromFile=True):
        "Load shader"
        shaderD = self.shaders[shaderName]
        shaderSource = shaderD[shaderType]
        if shaderType == "vertex":
            shader = QOpenGLShader(QOpenGLShader.Vertex)
        else:
            shader = QOpenGLShader(QOpenGLShader.Fragment)
        #
        if fromFile:
            isCompiled = shader.compileSourceFile(shaderSource)
        else:
            isCompiled = shader.compileSourceCode(shaderSource)
        #
        if isCompiled is False:
            print(shader.log())
            raise ValueError(
                "{0} shader {2} known as {1} is not compiled".format(
                    shaderType, shaderName, shaderSource
                )
            )
        return shader

    def loadVertexShader(self, shaderName: str, fromFile=True):
        "load vertex shader"
        return self.loadShader(shaderName, "vertex", fromFile)

    def loadFragmentShader(self, shaderName: str, fromFile=True):
        "load fragment shader"
        return self.loadShader(shaderName, "fragment", fromFile)

    def getGLInfo(self):
        "Get opengl info"
        info = "Vendor: {0}, Renderer: {1}, OpenGL Version: {2}, Shader Version: {3}".format(
            pygl.glGetString(pygl.GL_VENDOR),
            pygl.glGetString(pygl.GL_RENDERER),
            pygl.glGetString(pygl.GL_VERSION),
            pygl.glGetString(pygl.GL_SHADING_LANGUAGE_VERSION),
        )
        return info

    def moveCamera(self, direction: str):
        "Move camera to certain direction and update gl widget"
        self.camera.move(direction, deltaTime=0.05)
        pos = self.camera.position
        pos = self.limitMovement(pos)
        self.camera.setPosition(pos)
        self.update()

    def turnAround(self, x: float, y: float):
        ""
        self.camera.lookAround(xoffset=x, yoffset=y, pitchBound=True)
        self.update()

    def changeShininess(self, val: float):
        "set a new shininess value to cube fragment shader"
        self.shininess = val
        self.update()

    def moveLight(self, xoffset: float, yoffset: float, zoffset: float):
        "Translate light position vector to a new position"
        currentPos = self.lamp.position
        translationVec = QVector3D(xoffset, yoffset, zoffset)
        newpos = currentPos + translationVec
        newpos = self.limitMovement(newpos)
        self.lamp.setPosition(vec=newpos)
        self.update()

    def limitMovement(self, pos: QVector3D):
        "Limit position with respect to world limit"
        px = pos.x()
        py = pos.y()
        pz = pos.z()
        if px > self.worldLimitRight:
            pos.setX(self.worldLimitRight)
        elif px < self.worldLimitLeft:
            pos.setX(self.worldLimitLeft)
        if py > self.worldLimitTop:
            pos.setY(self.worldLimitTop)
        elif py < self.worldLimitBottom:
            pos.setY(self.worldLimitBottom)
        if pz > self.worldLimitDepthPos:
            pos.setZ(self.worldLimitDepthPos)
        elif pz < self.worldLimitDepthNeg:
            pos.setZ(self.worldLimitDepthNeg)
        return pos

    def rotateLight(self, xval: float, yval: float, zval: float):
        ""
        newRotVec = QVector3D(xval, yval, zval)
        self.rotVectorLamp = newRotVec
        self.lamp.setDirection(vec=newRotVec)
        self.update()

    def changeLampIntensity(self, channel: str, val: float):
        ""
        availables = ["red", "green", "blue"]
        if channel not in availables:
            mess = "Unknown channel name " + channel
            mess += ", available channels are: "
            mess += "red, green, blue"
            raise ValueError(mess)
        self.lamp.setIntensity(channel=channel, val=val)
        self.update()

    def changeLampIntensityCoefficient(self, channel: str, val: float):
        ""
        availables = ["red", "green", "blue"]
        if channel not in availables:
            mess = "Unknown channel name " + channel
            mess += ", available channels are: "
            mess += "red, green, blue"
            raise ValueError(mess)
        #
        self.lamp.setCoeffs(channel=channel, val=val)
        self.update()

    def changeAmbientCoeffs(self, val: float):
        self.ambientCoeff = val
        self.update()

    def setRotationAngle(self, val: float):
        self.rotationAngle = val
        self.update()

    def objectShader_init(self):
        "Object shader initialization"
        shader = "rgbptm"
        vshader = self.loadVertexShader(shader, fromFile=False)
        fshader = self.loadFragmentShader(shader, fromFile=False)
        self.program.addShader(vshader)
        self.program.addShader(fshader)
        for attr, adict in self.attrLoc.items():
            self.program.bindAttributeLocation(attr, adict["layout"])
        linked = self.program.link()
        print(shader, "shader linked: ", linked)

    def setObjectUniforms_proc(self):
        "set object shader uniforms"
        self.program.setUniformValue("lightPos", self.lamp.position)
        self.program.setUniformValue("coeffRed", self.lamp.color.x())
        self.program.setUniformValue("coeffGreen", self.lamp.color.y())
        self.program.setUniformValue("coeffBlue", self.lamp.color.z())

    def lampShader_init(self):
        "Lamp shader initialization"
        shader = "lamp"
        vshader = self.loadVertexShader(shader, fromFile=False)
        fshader = self.loadFragmentShader(shader, fromFile=False)
        self.lampProgram.addShader(vshader)
        self.lampProgram.addShader(fshader)
        self.lampProgram.bindAttributeLocation(
            "aPos", self.attrLoc["aPos"]["layout"]
        )

        # lamp needs:
        # projection, view, model
        linked = self.lampProgram.link()
        print("lamp shader linked: ", linked)

    def setLampShaderUniforms(self):
        "set lamp shader uniforms"
        projection = QMatrix4x4()
        projection.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        view = self.camera.getViewMatrix()
        model = QMatrix4x4()
        model.translate(self.lamp.position)
        model.rotate(self.rotationAngle, self.rotVectorLamp)
        self.lampProgram.setUniformValue("projection", projection)
        self.lampProgram.setUniformValue("view", view)
        self.lampProgram.setUniformValue("model", model)
        self.lampProgram.setUniformValue("lightColor", self.lamp.color)

    def cleanUpGL(self):
        "Clean up everything"
        self.context.makeCurrent()
        self.releaseTextures_proc()
        del self.program
        del self.lampProgram
        self.program = None
        self.lampProgram = None
        self.doneCurrent()

    def resizeGL(self, width: int, height: int):
        "Resize the viewport"
        funcs = self.context.functions()
        funcs.glViewport(0, 0, width, height)

    def initializeGL(self):
        # create context and make it current
        self.context.create()
        self.context.aboutToBeDestroyed.connect(self.cleanUpGL)

        # initialize functions
        funcs = self.context.functions()
        funcs.initializeOpenGLFunctions()
        funcs.glClearColor(0.0, 0.0, 0.0, 0)
        funcs.glEnable(pygl.GL_DEPTH_TEST)
        funcs.glEnable(pygl.GL_CULL_FACE)

        # load shaders: lamp shader, object shader
        # initialize shaders
        self.program = QOpenGLShaderProgram(self.context)
        self.objectShader_init()
        print("object init done")
        isb = self.program.bind()
        print("object shader bound: ", isb)
        # set uniforms to shaders
        floatSize = ctypes.sizeof(ctypes.c_float)
        self.vbo.create()
        self.vbo.bind()
        self.vbo.allocate(
            self.vertices.tobytes(), self.vertices.size * floatSize
        )
        self.vbo.bind()
        for aname, aprop in self.attrLoc.items():
            self.program.enableAttributeArray(aprop["layout"])
            self.program.setAttributeBuffer(
                aprop["layout"],
                pygl.GL_FLOAT,
                self.attrLoc["aPos"]["offset"],  # 0
                self.attrLoc["aPos"]["stride"],  # tuple Size: vec3
                self.attrLoc["aPos"]["stride"] * floatSize,
            )

        self.lampProgram = QOpenGLShaderProgram(self.context)
        self.lampShader_init()
        print("lamp init done")
        # set vao and vbo for shaders
        isb = self.lampProgram.bind()
        print("lamp shader bound: ", isb)

        floatSize = ctypes.sizeof(ctypes.c_float)
        # lamp vbo
        self.lampVbo.create()
        self.lampVbo.bind()
        self.lampVbo.allocate(
            self.lampVertices.tobytes(), self.lampVertices.size * floatSize
        )
        self.lampVbo.bind()
        self.lampProgram.enableAttributeArray(self.attrLoc["aPos"]["layout"])
        self.lampProgram.setAttributeBuffer(
            self.attrLoc["aPos"]["layout"],  # layout location 0
            pygl.GL_FLOAT,
            self.attrLoc["aPos"]["offset"],  # 0
            self.attrLoc["aPos"]["stride"],  # tuple Size: vec3
            self.attrLoc["aPos"]["stride"] * floatSize,
        )

    def paintGL(self):
        "drawing loop"
        funcs = self.context.functions()
        # clean up what was drawn
        funcs.glClear(pygl.GL_COLOR_BUFFER_BIT | pygl.GL_DEPTH_BUFFER_BIT)
        # bind shader: object
        self.program.bind()
        # set uniforms to shader
        self.setObjectUniforms_proc()
        # bind texture
        self.vbo.bind()
        self.program.bind()
        funcs.glDrawElements(pygl.GL_POINTS, self.indices.size,
                pygl.GL_UNSIGNED_INT, self.indices.tobytes())

        # end render viewer
        # self.releaseTextures_proc()
        self.vbo.release()
        self.program.release()

        # bind shader: lamp
        self.lampProgram.bind()
        # self.lampVbo.bind()
        # set uniforms to shader
        self.setLampShaderUniforms()
        # render lamp
        funcs.glDrawArrays(pygl.GL_TRIANGLES, 0, 36)
        self.lampVbo.release()
        self.lampProgram.release()
        # self.program.bind()
        # self.renderViewer()
