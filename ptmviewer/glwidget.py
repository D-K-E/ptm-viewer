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


from PySide2.shiboken2 import VoidPtr
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


class AbstractPtmGLWidget(QOpenGLWidget):
    "OpenGL widget"

    def __init__(self, parent=None):
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
        self.attrLoc = {}
        self.rowsize = 0
        # opengl data

        self.context = QOpenGLContext()
        self.lampVao = QOpenGLVertexArrayObject()
        self.vao = QOpenGLVertexArrayObject()
        self.lampVbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)

        self.program = QOpenGLShaderProgram()
        self.lampProgram = QOpenGLShaderProgram()
        # vertices
        self.vertices = None
        self.indices = None

        # texture
        self.texture = None

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
        self.worldLimitTop = 3.0  # + imTopY
        self.worldLimitBottom = -3.0  # + imBottomY
        self.worldLimitRight = 3.0  # + imRightX
        self.worldLimitLeft = -3.0  # + imLeftX
        self.worldLimitDepthNeg = -20.0
        self.worldLimitDepthPos = 20.0

    def setAttrLocFromShader(self, shaderName: str, shaderD: dict) -> None:
        "Set attribute location dict from shader dict"
        self.attrLoc[shaderName] = shaderD["attribute_info"]

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

    def linkShaders(
        self, shaderName: str, program: QOpenGLShaderProgram, fromFile=True
    ):
        "Link shaders to program"
        shaderD = self.shaders[shaderName]
        self.setAttrLocFromShader(shaderName, shaderD)
        self.setStride(shaderName)
        vshader = self.loadVertexShader(shaderName, fromFile)
        fshader = self.loadFragmentShader(shaderName, fromFile)
        program.addShader(vshader)
        program.addShader(fshader)
        # bind attribute locations
        for aname, aparams in self.attrLoc[shaderName].items():
            if aname != "stride":
                print("attribute name", aname)
                print("attribute params:", aparams)
                program.bindAttributeLocation(aname, aparams["layout"])
        linked = program.link()
        if not linked:
            print("program linked:", linked)
            print("failer log:", program.log())
        #
        return program

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

    def changeAmbientCoeffs(self, val: float):
        self.ambientCoeff = val
        self.update()

    def setRotationAngle(self, val: float):
        self.rotationAngle = val
        self.update()

    def createTexture(self, img: QImage, unit: int):
        "create texture"
        texture = QOpenGLTexture(QOpenGLTexture.Target2D)
        texture.create()
        texture.bind(unit)
        texture.setData(img)
        texture.setMinMagFilters(
            QOpenGLTexture.Nearest, QOpenGLTexture.Nearest
        )
        texture.setWrapMode(QOpenGLTexture.DirectionS, QOpenGLTexture.Repeat)
        texture.setWrapMode(QOpenGLTexture.DirectionT, QOpenGLTexture.Repeat)
        return texture

    def setStride(self, shaderName: str) -> None:
        "set row size for vertex array buffer"
        stride = 0
        for aName, aprop in self.attrLoc[shaderName].items():
            if isinstance(aprop, dict):
                stride += aprop["size"]
        self.attrLoc[shaderName]["stride"] = stride

    def cleanUpGL(self):
        "Clean up everything"
        self.context.makeCurrent()
        del self.program
        self.program = None
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
        # create shader program
        self.program = QOpenGLShaderProgram(self.context)
        # add shaders to program
        shaderName = ""
        self.program = self.linkShaders(
            shaderName, program=self.program, fromFile=False
        )

        isb = self.program.bind()
        self.vbo.create()
        self.vbo.bind()
        self.vao.create()
        isVaoBind = self.vao.bind()
        print("vao bound:", isVaoBind)
        #
        self.vbo.create()
        isVboBind = self.vbo.bind()
        print("vbo bound:", isVboBind)
        #
        floatSize = ctypes.sizeof(ctypes.c_float)
        # allocate buffer
        self.vbo.allocate(
            self.vertices.tobytes(), self.vertices.size * floatSize
        )
        # enable given attribute and attribute pointer
        attrName = ""
        rowsize = 1
        funcs.glEnableVertexAttribArray(self.attrLoc[attrName]["layout"])
        funcs.glVertexAttribPointer(
            self.attrLoc[attrName]["layout"],
            self.attrLoc[attrName]["size"],
            int(pygl.GL_FLOAT),
            int(pygl.GL_FALSE),
            rowsize * floatSize,
            VoidPtr(self.attrLoc[attrName]["offset"] * floatSize),
        )
        # activate texture
        self.texture = self.createTexture()
        # release ressources to bind other objects to opengl
        self.vbo.release()
        self.vao.release()
        self.program.release()

        # create shader program
        self.lampProgram = QOpenGLShaderProgram(self.context)
        # add shaders to program
        shaderName = ""
        self.lampProgram = self.linkShaders(
            shaderName, program=self.lampProgram, fromFile=False
        )

        self.lampProgram.bind()
        # do the same for other vaos and vbos
        self.lampVao.create()
        isVao = self.lampVao.bind()
        print("lamp vao bound:", isVao)

        self.lampVbo.create()
        isVbo = self.lampVbo.bind()
        print("lamp vbo bound:", isVbo)
        data = np.array([], dtype=ctypes.c_float)
        self.lampVbo.allocate(data.tobytes(), data.size * floatSize)
        # enable attribute and attribute pointer
        funcs.glEnableVertexAttribArray(self.attrLoc[attrName]["layout"])
        funcs.glVertexAttribPointer(
            self.attrLoc[attrName]["layout"],
            self.attrLoc[attrName]["size"],
            int(pygl.GL_FLOAT),
            int(pygl.GL_FALSE),
            rowsize * floatSize,
            VoidPtr(self.attrLoc[attrName]["offset"] * floatSize),
        )
        # release ressources to bind other objects to open gl
        self.lampVbo.release()
        self.lampVao.release()
        self.lampProgram.release()

    def paintGL(self):
        "drawing loop"
        funcs = self.context.functions()
        # clean up what was drawn
        funcs.glClear(pygl.GL_COLOR_BUFFER_BIT | pygl.GL_DEPTH_BUFFER_BIT)
        # bind shader: object, use shader
        self.program.bind()
        # bind vao, vbo
        self.vao.bind()
        # bind texture
        self.texture.bind()
        # render everything
        vboData = np.array([], dtype=ctypes.c_float)
        funcs.glDrawArrays(pygl.GL_TRIANGLES, 0, vboData.size)


class PtmLambertianGLWidget(AbstractPtmGLWidget):
    "OpenGL widget that displays ptm diffuse map"

    def __init__(self, ptmImage: QImage, parent=None):
        super().__init__(parent)
        self.img = ptmImage.mirrored()
        self.texUnit = 0
        self.texture = None

        # fmt: off
        self.vertices = np.array(
            [
                # position       # normal       # texture coordinates
                0.7, 0.7, -3.0, 0.0, 0.0, -1.0, 1.0, 1.0,  # top right
                0.7, -0.7, -3.0, 0.0, 0.0, -1.0, 1.0, 0.0,  # bottom right
                -0.7, -0.7, -3.0, 0.0, 0.0, -1.0, 0.0, 0.0,  # bottom left

                -0.7, 0.7, -3.0, 0.0, 0.0, -1.0, 0.0, 1.0,  # top left
                0.7, 0.7, -3.0, 0.0, 0.0, -1.0, 1.0, 1.0,  # top right
                -0.7, -0.7, -3.0, 0.0, 0.0, -1.0, 0.0, 0.0,  # bottom left
            ],
            dtype=ctypes.c_float,
        )
        # fmt: on

    def cleanUpGL(self):
        self.context.makeCurrent()
        del self.program
        del self.lampProgram
        self.program = None
        self.lampProgram = None
        self.texture.destroy()
        self.vbo.destroy()
        self.lampVbo.destroy()
        self.doneCurrent()

    def setShaderUniforms(self):
        "set shader uniforms"
        projectionMatrix = QMatrix4x4()
        projectionMatrix.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        viewMatrix = self.camera.getViewMatrix()
        model = QMatrix4x4()
        color = self.lamp.color
        pos = self.lamp.position
        self.program.setUniformValue("projection", projectionMatrix)
        self.program.setUniformValue("view", viewMatrix)
        self.program.setUniformValue("model", model)
        self.program.setUniformValue("lightColor", color)
        self.program.setUniformValue("lightPos", pos)
        # self.program.setUniformValue("diffuseMap", self.texUnit)

    def setLampShaderUniforms_proc(self):
        "set lamp shader uniforms in paintgl"
        projectionMatrix = QMatrix4x4()
        projectionMatrix.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        viewMatrix = self.camera.getViewMatrix()
        lampModel = QMatrix4x4()
        lampModel.translate(self.lamp.position)
        lampModel.rotate(self.rotationAngle, self.rotVectorLamp)
        self.lampProgram.setUniformValue("projection", projectionMatrix)
        self.lampProgram.setUniformValue("view", viewMatrix)
        self.lampProgram.setUniformValue("model", lampModel)
        self.lampProgram.setUniformValue("lightColor", self.lamp.color)

    def initializeGL(self):
        "Initialize gl"
        self.context.create()
        self.context.aboutToBeDestroyed.connect(self.cleanUpGL)

        # initialize functions
        funcs = self.context.functions()
        funcs.initializeOpenGLFunctions()
        funcs.glClearColor(0.0, 0.4, 0.4, 0)
        funcs.glEnable(pygl.GL_DEPTH_TEST)
        funcs.glEnable(pygl.GL_TEXTURE_2D)

        floatSize = ctypes.sizeof(ctypes.c_float)
        # create lamp shader
        self.lampProgram = QOpenGLShaderProgram(self.context)
        shname = "lamp"
        vshader = self.loadVertexShader(shname, fromFile=False)
        fshader = self.loadFragmentShader(shname, fromFile=False)
        shaderD = self.shaders[shname]
        self.setAttrLocFromShader(shname, shaderD)
        self.setStride(shname)
        self.lampProgram.addShader(vshader)
        self.lampProgram.addShader(fshader)
        self.lampProgram.bindAttributeLocation("aPos", 0)
        linked = self.lampProgram.link()
        if not linked:
            print("program linked:", linked)
            print("failer log:", self.lampProgram.log())

        self.lampProgram.bind()

        # create object shader
        self.program = QOpenGLShaderProgram(self.context)
        shname = "lambert"
        vshader = self.loadVertexShader(shname, fromFile=False)
        fshader = self.loadFragmentShader(shname, fromFile=False)
        shaderD = self.shaders[shname]
        self.setAttrLocFromShader(shname, shaderD)
        self.setStride(shname)
        self.program.addShader(vshader)
        self.program.addShader(fshader)
        self.program.bindAttributeLocation("aPos", 0)
        self.program.bindAttributeLocation("aNormal", 1)
        self.program.bindAttributeLocation("aTexCoord", 2)
        linked = self.program.link()
        if not linked:
            print("program linked:", linked)
            print("failer log:", self.program.log())
        #
        self.program.bind()
        self.program.setUniformValue("diffuseMap", self.texUnit)

        # lamp: vbo
        self.lampVbo.create()
        self.lampVbo.bind()
        self.lampVbo.allocate(
            self.lampVertices.tobytes(), self.lampVertices.size * floatSize
        )
        print("lamp vbo size:", self.lampVbo.size())

        # lamp: vao
        self.lampVao.create()
        self.lampVao.bind()
        # position
        attrName = "aPos"
        shname = "lamp"
        stride = self.attrLoc[shname]["stride"] * floatSize
        layout = self.attrLoc[shname][attrName]["layout"]
        size = self.attrLoc[shname][attrName]["size"]
        offset = self.attrLoc[shname][attrName]["offset"] * floatSize
        funcs.glEnableVertexAttribArray(layout)
        funcs.glVertexAttribPointer(
            layout,
            size,
            int(pygl.GL_FLOAT),
            int(pygl.GL_FALSE),
            stride,
            VoidPtr(offset),
        )
        # end lamp: vao, vbo

        # object: vbo, vao
        self.vbo.create()
        self.vbo.bind()
        self.vbo.allocate(
            self.vertices.tobytes(), self.vertices.size * floatSize
        )
        print("vbo size:", self.vbo.size())
        #
        self.vao.create()
        self.vao.bind()
        #
        shname = "lambert"
        stride = self.attrLoc[shname]["stride"] * floatSize
        # position
        attrName = "aPos"
        layout = self.attrLoc[shname][attrName]["layout"]
        size = self.attrLoc[shname][attrName]["size"]
        offset = self.attrLoc[shname][attrName]["offset"] * floatSize
        funcs.glEnableVertexAttribArray(layout)
        funcs.glVertexAttribPointer(
            layout,
            size,
            int(pygl.GL_FLOAT),
            int(pygl.GL_FALSE),
            stride,
            VoidPtr(offset),
        )
        #
        # normal
        attrName = "aNormal"
        layout = self.attrLoc[shname][attrName]["layout"]
        size = self.attrLoc[shname][attrName]["size"]
        offset = self.attrLoc[shname][attrName]["offset"] * floatSize
        funcs.glEnableVertexAttribArray(layout)
        funcs.glVertexAttribPointer(
            layout,
            size,
            int(pygl.GL_FLOAT),
            int(pygl.GL_FALSE),
            stride,
            VoidPtr(offset),
        )
        #
        # texture coord
        attrName = "aTexCoord"
        layout = self.attrLoc[shname][attrName]["layout"]
        size = self.attrLoc[shname][attrName]["size"]
        offset = self.attrLoc[shname][attrName]["offset"] * floatSize
        funcs.glEnableVertexAttribArray(layout)
        funcs.glVertexAttribPointer(
            layout,
            size,
            int(pygl.GL_FLOAT),
            int(pygl.GL_FALSE),
            stride,
            VoidPtr(offset),
        )
        #
        # create texture
        self.texture = self.createTexture(img=self.img, unit=self.texUnit)
        self.lampVbo.release()
        self.vbo.release()
        self.vao.release()

    def paintGL(self):
        "paint gl drawing loop"
        funcs = self.context.functions()
        funcs.glClear(pygl.GL_COLOR_BUFFER_BIT | pygl.GL_DEPTH_BUFFER_BIT)
        #
        # render object
        self.vao.bind()
        self.program.bind()
        self.setShaderUniforms()
        self.texture.bind()
        funcs.glDrawArrays(pygl.GL_TRIANGLES, 0, 6)

        # render lamp
        self.lampVao.bind()
        self.lampProgram.bind()
        self.setLampShaderUniforms_proc()
        funcs.glDrawArrays(pygl.GL_TRIANGLES, 0, 36)
