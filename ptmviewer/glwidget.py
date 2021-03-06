# author: Kaan Eraslan

# Purpose: Application wrapper for ptm viewer

# glwidget class

import sys
import ctypes
import numpy as np
from typing import Tuple, List

import pdb

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
from ptmviewer.utils.camera import QtCamera
from ptmviewer.utils.light import QtShaderLight
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
    messageBox.setDetailedText("Run:\npip install PyOpenGL PyOpenGL_accelerate")
    messageBox.exec_()
    sys.exit(1)

from PySide2.QtWidgets import QOpenGLWidget


class AbstractGLWidgetHelper(QOpenGLWidget):
    "OpenGL widget"

    def __init__(self, parent=None):
        QOpenGLWidget.__init__(self, parent)
        self.attrLoc = {}

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

    def getGLInfo(self):
        "Get opengl info"
        info = "Vendor: {0}, Renderer: {1}, OpenGL Version: {2}, Shader Version: {3}".format(
            pygl.glGetString(pygl.GL_VENDOR),
            pygl.glGetString(pygl.GL_RENDERER),
            pygl.glGetString(pygl.GL_VERSION),
            pygl.glGetString(pygl.GL_SHADING_LANGUAGE_VERSION),
        )
        return info

    def createTexture(self, img: QImage, unit: int):
        "create texture"
        texture = QOpenGLTexture(QOpenGLTexture.Target2D)
        texture.create()
        texture.bind(unit)
        texture.setData(img)
        texture.setMinMagFilters(QOpenGLTexture.Nearest, QOpenGLTexture.Nearest)
        texture.setMinMagFilters(QOpenGLTexture.Nearest, QOpenGLTexture.Nearest)
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


class UiGLEvents(AbstractGLWidgetHelper):
    "Deal with user interface events"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # camera
        self.camera = QtCamera()
        self.camera.position = QVector3D(0.0, 0.0, 3.0)
        self.camera.front = QVector3D(0.0, 0.0, -1.0)
        self.camera.up = QVector3D(0.0, 1.0, 0.0)
        self.camera.movementSensitivity = 0.05

        # light source: point light
        self.lamp = QtShaderLight()
        self.shininess = 30.0
        # dealing with rotation and axes
        self.rotationAngle = 45.0
        self.cameraRotationAngle = 45.0
        self.rotationAxes = ["x"]

    def set_rotate_axes(self, axes: List[QVector3D]):
        self.rotationAxes = axes

    # lamp methods
    def move_light(self, direction: str):
        "Translate light position vector to a new position"
        self.lamp.move(direction, deltaTime=0.05)
        pos = self.lamp.position
        newpos = self.limit_movement(pos)
        self.lamp.set_position(newpos)
        self.update()

    def set_euler_angle_to_lamp(self, angle: float):
        "set euler angles to lamp with available axes"
        for axis in self.rotationAxes:
            if axis == "x":
                self.lamp.set_roll(angle)
            elif axis == "y":
                self.lamp.set_pitch(angle)
            elif axis == "z":
                self.lamp.set_yaw(angle)
        self.update()

    def change_lamp_diffuse_intensity(self, channel: str, val: float):
        ""
        availables = ["red", "green", "blue"]
        if channel not in availables:
            mess = "Unknown channel name " + channel
            mess += ", available channels are: "
            mess += "red, green, blue"
            raise ValueError(mess)
        self.lamp.set_channel_intensity(channel=channel, val=val, lsource="diffuse")
        self.update()

    def change_lamp_specular_intensity(self, channel: str, val: float):
        ""
        availables = ["red", "green", "blue"]
        if channel not in availables:
            mess = "Unknown channel name " + channel
            mess += ", available channels are: "
            mess += "red, green, blue"
            raise ValueError(mess)
        self.lamp.set_channel_intensity(channel=channel, val=val, lsource="specular")
        self.update()

    def change_lamp_ambient_intensity(self, channel: str, val: float):
        ""
        availables = ["red", "green", "blue"]
        if channel not in availables:
            mess = "Unknown channel name " + channel
            mess += ", available channels are: "
            mess += "red, green, blue"
            raise ValueError(mess)
        self.lamp.set_channel_intensity(channel=channel, val=val, lsource="ambient")
        self.update()

    def change_lamp_diffuse_coefficient(self, channel: str, val: float):
        "lamp diffuse coefficient"
        availables = ["red", "green", "blue"]
        if channel not in availables:
            mess = "Unknown channel name " + channel
            mess += ", available channels are: "
            mess += "red, green, blue"
            raise ValueError(mess)
        self.lamp.set_channel_coeff(channel=channel, val=val, lsource="diffuse")
        self.update()

    def change_lamp_specular_coefficient(self, channel: str, val: float):
        "lamp diffuse coefficient"
        availables = ["red", "green", "blue"]
        if channel not in availables:
            mess = "Unknown channel name " + channel
            mess += ", available channels are: "
            mess += "red, green, blue"
            raise ValueError(mess)
        self.lamp.set_channel_coeff(channel=channel, val=val, lsource="specular")
        self.update()

    def change_lamp_ambient_coefficient(self, channel: str, val: float):
        "lamp diffuse coefficient"
        availables = ["red", "green", "blue"]
        if channel not in availables:
            mess = "Unknown channel name " + channel
            mess += ", available channels are: "
            mess += "red, green, blue"
            raise ValueError(mess)
        self.lamp.set_channel_coeff(channel=channel, val=val, lsource="ambient")
        self.update()

    # camera methods
    def move_camera(self, direction: str):
        "Move camera to certain direction and update gl widget"
        self.camera.move(direction, deltaTime=0.05)
        pos = self.camera.position
        pos = self.limit_movement(pos)
        self.camera.set_position(pos)
        self.update()

    def rotate_camera_model(self, model: QMatrix4x4):
        "rotate camera model"
        for axis in self.rotationAxes:
            model.rotate(self.cameraRotationAngle, axis)
        return model

    def rotate_camera(self, angle: float):
        self.cameraRotationAngle = angle
        self.update()

    def set_euler_angles_to_camera(self, angle: float):
        "set euler angles to camera"
        for axis in self.rotationAxes:
            if axis == "x":
                self.camera.set_roll(angle)
            elif axis == "y":
                self.camera.set_pitch(angle)
            elif axis == "z":
                self.camera.set_yaw(angle)
        self.update()

    def turn_camera_around(self, x: float, y: float):
        ""
        self.camera.lookAround(xoffset=x, yoffset=y, pitchBound=True)
        self.update()


class AbstractPointLightPtmGLWidget(UiGLEvents):
    "OpenGL widget"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # shaders
        self.shaders = shaders
        self.rowsize = 0
        self.lampShaderName = ""
        self.objectShaderName = ""

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
        self.textures = [
            # {"unit": 0,
            # "data": QImage,
            # "name": str,
            # "texture": QOpenGLTexture}
        ]

        # fmt: off
        self.lampVertices = np.array(
            [  # first square for cube
                -0.05,  # first corner
                -0.05,
                -0.05,
                0.05,  # second corner
                -0.05,
                -0.05,
                0.05,  # third corner
                0.05,
                -0.05,
                0.05,  # fourth corner
                0.05,
                -0.05,
                # second square for cube
                -0.05,
                0.05,
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                0.05,
                0.05,
                -0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                -0.05,
                0.05,
                0.05,
                -0.05,
                -0.05,
                0.05,
                -0.05,
                0.05,
                0.05,
                -0.05,
                0.05,
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                0.05,
                -0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                -0.05,
                0.05,
                -0.05,
                -0.05,
                0.05,
                -0.05,
                -0.05,
                0.05,
                -0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                -0.05,
                -0.05,
                -0.05,
                0.05,
                -0.05,
                -0.05,
                0.05,
                -0.05,
                0.05,
                0.05,
                -0.05,
                0.05,
                -0.05,
                -0.05,
                0.05,
                -0.05,
                -0.05,
                -0.05,
                -0.05,
                0.05,
                -0.05,
                0.05,
                0.05,
                -0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                -0.05,
                0.05,
                0.05,
                -0.05,
                0.05,
                -0.05,
            ],
            dtype=ctypes.c_float,
        )
        # fmt: on
        #
        self.worldLimitTop = 3.0  # + imTopY
        self.worldLimitBottom = -3.0  # + imBottomY
        self.worldLimitRight = 3.0  # + imRightX
        self.worldLimitLeft = -3.0  # + imLeftX
        self.worldLimitDepthNeg = -20.0
        self.worldLimitDepthPos = 20.0

    def limit_movement(self, pos: QVector3D):
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

    def change_shininess(self, val: float):
        "set a new shininess value to cube fragment shader"
        self.shininess = val
        self.update()

    def createTextures(self):
        "create textures using create texture and a texture list"
        for texDict in self.textures:
            texDict["texture"] = self.createTexture(
                img=texDict["data"], unit=texDict["unit"]
            )

    def lampShader_init(self, shname: str):
        "lamp shader initialization"
        vshader = self.loadVertexShader(shname, fromFile=False)
        fshader = self.loadFragmentShader(shname, fromFile=False)
        shaderD = self.shaders[shname]
        self.setAttrLocFromShader(shname, shaderD)
        self.setStride(shname)
        self.lampProgram.addShader(vshader)
        self.lampProgram.addShader(fshader)
        for aname, adict in self.attrLoc[shname].items():
            if aname != "stride":
                layout = adict["layout"]
                self.lampProgram.bindAttributeLocation(aname, layout)
        #
        linked = self.lampProgram.link()
        if not linked:
            print("program linked:", linked)
            print("failer log:", self.lampProgram.log())

    def programShader_init(self, shname: str):
        "Initialize program shader"
        vshader = self.loadVertexShader(shname, fromFile=False)
        fshader = self.loadFragmentShader(shname, fromFile=False)
        shaderD = self.shaders[shname]
        self.setAttrLocFromShader(shname, shaderD)
        self.setStride(shname)
        self.program.addShader(vshader)
        self.program.addShader(fshader)
        for aname, adict in self.attrLoc[shname].items():
            if aname != "stride":
                layout = adict["layout"]
                self.program.bindAttributeLocation(aname, layout)
        #
        linked = self.program.link()
        if not linked:
            print("program linked:", linked)
            print("failer log:", self.program.log())

    def programShader_init_uniforms(self):
        "set uniforms during initilization"
        pass

    def lampShader_init_uniforms(self):
        "set uniforms during initilization of lamp shader"
        pass

    def setLampShaderUniforms_proc(self):
        "Set lamp shader uniforms"
        pass

    def setObjectShaderUniforms_proc(self):
        ""
        pass

    def bindTextures_proc(self):
        "bind textures to paint event"
        for texdict in self.textures:
            texdict["texture"].bind()

    def cleanUpGL(self):
        "Clean up everything"
        self.context.makeCurrent()
        del self.program
        del self.lampProgram
        self.program = None
        self.lampProgram = None
        self.vbo.destroy()
        self.lampVbo.destroy()
        self.lampVao.destroy()
        self.vao.destroy()
        [tdict["texture"].destroy() for tdict in self.textures]
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
        funcs.glClearColor(0.0, 0.4, 0.4, 0)
        funcs.glEnable(pygl.GL_DEPTH_TEST)
        funcs.glEnable(pygl.GL_TEXTURE_2D)

        # vars to be used in initialization
        floatSize = ctypes.sizeof(ctypes.c_float)
        # create lamp shader
        self.lampProgram = QOpenGLShaderProgram(self.context)
        self.lampShader_init(self.lampShaderName)
        self.lampProgram.bind()
        self.lampShader_init_uniforms()

        # create object shader
        self.program = QOpenGLShaderProgram(self.context)
        self.programShader_init(self.objectShaderName)
        self.program.bind()
        self.programShader_init_uniforms()

        # lamp: vbo
        self.lampVbo.create()
        self.lampVbo.bind()
        self.lampVbo.allocate(
            self.lampVertices.tobytes(), self.lampVertices.size * floatSize
        )

        # lamp: vao
        self.lampVao.create()
        self.lampVao.bind()
        stride = self.attrLoc[self.lampShaderName]["stride"] * floatSize
        # position
        for attrName, adict in self.attrLoc[self.lampShaderName].items():
            if attrName != "stride":
                layout = adict["layout"]
                size = adict["size"]
                offset = adict["offset"] * floatSize
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
        self.lampVao.release()
        self.lampVbo.release()

        # object: vbo, vao
        self.vbo.create()
        self.vbo.bind()
        self.vbo.allocate(self.vertices.tobytes(), self.vertices.size * floatSize)
        #
        self.vao.create()
        self.vao.bind()
        #
        stride = self.attrLoc[self.objectShaderName]["stride"] * floatSize
        for attrName, adict in self.attrLoc[self.objectShaderName].items():
            if attrName != "stride":
                layout = adict["layout"]
                size = adict["size"]
                offset = adict["offset"] * floatSize
                funcs.glEnableVertexAttribArray(layout)
                funcs.glVertexAttribPointer(
                    layout,
                    size,
                    int(pygl.GL_FLOAT),
                    int(pygl.GL_FALSE),
                    stride,
                    VoidPtr(offset),
                )
        # end object: vbo, vao
        self.vbo.release()
        self.vao.release()
        # create texture
        self.createTextures()

    def paintGL(self):
        "paint gl drawing loop"
        funcs = self.context.functions()
        funcs.glClear(pygl.GL_COLOR_BUFFER_BIT | pygl.GL_DEPTH_BUFFER_BIT)
        #
        # render object
        self.vao.bind()
        self.program.bind()
        self.setObjectShaderUniforms_proc()
        self.bindTextures_proc()
        funcs.glDrawArrays(pygl.GL_TRIANGLES, 0, 6)

        # render lamp
        self.lampVao.bind()
        self.lampProgram.bind()
        self.setLampShaderUniforms_proc()
        funcs.glDrawArrays(pygl.GL_TRIANGLES, 0, 36)


class PtmLambertianGLWidget(AbstractPointLightPtmGLWidget):
    "OpenGL widget that displays ptm diffuse map"

    def __init__(
        self,
        ptmImage: QImage,
        lampShaderName="lamp",
        objectShaderName="lambert",
        parent=None,
    ):
        super().__init__(parent)
        self.textures = [
            {
                "unit": 0,
                "name": "diffuseMap",
                "data": ptmImage.mirrored(),
                "texture": None,
            }
        ]
        self.textures = [
            {
                "unit": 0,
                "name": "diffuseMap",
                "data": ptmImage.mirrored(),
                "texture": None,
            }
        ]
        self.lampShaderName = lampShaderName
        self.objectShaderName = objectShaderName

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

    def setObjectShaderUniforms_proc(self):
        "set shader uniforms"
        projectionMatrix = QMatrix4x4()
        projectionMatrix.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        viewMatrix = self.camera.getViewMatrix()
        model = QMatrix4x4()
        color = self.lamp.diffuse.color
        pos = self.lamp.position
        # pdb.set_trace()
        self.program.setUniformValue("projection", projectionMatrix)
        self.program.setUniformValue("view", viewMatrix)
        self.program.setUniformValue("model", model)
        self.program.setUniformValue("lightColor", color)
        self.program.setUniformValue("lightPos", pos)

    def programShader_init_uniforms(self):
        "set texture uniforms for initialization"
        for texdict in self.textures:
            self.program.setUniformValue(texdict["name"], texdict["unit"])

    def setLampShaderUniforms_proc(self):
        "set lamp shader uniforms in paintgl"
        projectionMatrix = QMatrix4x4()
        projectionMatrix.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        viewMatrix = self.camera.getViewMatrix()
        lampModel = self.lamp.get_model_matrix()
        # pdb.set_trace()
        self.lampProgram.setUniformValue("projection", projectionMatrix)
        self.lampProgram.setUniformValue("view", viewMatrix)
        self.lampProgram.setUniformValue("model", lampModel)
        self.lampProgram.setUniformValue("lightColor", self.lamp.diffuse.color)


class PtmNormalMapGLWidget(PtmLambertianGLWidget):
    "Opengl widget that displays ptm diffuse map and a normal map"

    def __init__(
        self,
        ptmImage: QImage,
        normalMap: QImage,
        objectShaderName="quad",
        lampShaderName="lamp",
        parent=None,
    ):
        super().__init__(ptmImage, parent)
        self.objectShaderName = objectShaderName
        self.lampShaderName = lampShaderName
        self.textures = [
            {
                "texture": None,
                "name": "diffuseMap",
                "unit": 0,
                "data": ptmImage.mirrored(),
            },
            {
                "texture": None,
                "name": "normalMap",
                "unit": 1,
                "data": normalMap.mirrored(),
            },
        ]
        self.coords = {
            "pos": {
                "tl": QVector3D(-1.0, 1.0, 0.0),
                "bl": QVector3D(-1.0, -1.0, 0.0),
                "br": QVector3D(1.0, -1.0, 0.0),
                "tr": QVector3D(1.0, 1.0, 0.0),
            },
            "uv": {
                "tl": QVector2D(0.0, 1.0),
                "bl": QVector2D(0.0, 0.0),
                "br": QVector2D(1.0, 0.0),
                "tr": QVector2D(1.0, 1.0),
            },
            "n": QVector3D(0.0, 0.0, 1.0),
        }
        self.vertices = None
        self.fromCoords2Vertices()
        # fmt: off
        # vertices = [
        # first triangle
        # self.coords["pos"][0].x(), self.coords["pos"][0].y(),
        # self.coords["pos"][0].z(), self.coords["n"].x(),
        # self.coords["n"].y(), self.coords["n"].z(), t1x, t1y, t1z,
        # bit1x, bit1y, bit1z,
        # self.coords["pos"][1].x(), self.coords["pos"][1].y(),
        # self.coords["pos"][1].z(), self.coords["n"].x(),
        # self.coords["n"].y(), self.coords["n"].z(), t1x, t1y, t1z,
        # bit1x, bit1y, bit1z,
        # self.coords["pos"][2].x(), self.coords["pos"][2].y(),
        # self.coords["pos"][2].z(), self.coords["n"].x(),
        # self.coords["n"].y(), self.coords["n"].z(), t1x, t1y, t1z,
        # bit1x, bit1y, bit1z,

        # second triangle
        # self.coords["pos"][0].x(), self.coords["pos"][0].y(),
        # self.coords["pos"][0].z(), self.coords["n"].x(),
        # self.coords["n"].y(), self.coords["n"].z(), t2x, t2y, t2z,
        # bit2x, bit2y, bit2z,
        # self.coords["pos"][2].x(), self.coords["pos"][2].y(),
        # self.coords["pos"][2].z(), self.coords["n"].x(),
        # self.coords["n"].y(), self.coords["n"].z(), t2x, t2y, t2z,
        # bit2x, bit2y, bit2z,
        # self.coords["pos"][3].x(), self.coords["pos"][3].y(),
        # self.coords["pos"][3].z(), self.coords["n"].x(),
        # self.coords["n"].y(), self.coords["n"].z(), t2x, t2y, t2z,
        # bit2x, bit2y, bit2z
        # ]
        # fmt: on

    def computeTangentBiTangent(
        self,
        edge1: QVector3D,
        edge2: QVector3D,
        deltaUv1: QVector3D,
        deltaUv2: QVector3D,
    ):
        "compute tangent and bitangent vectors given params"
        t1 = deltaUv1.x() * deltaUv2.y()
        t2 = deltaUv1.y() * deltaUv2.x()
        t3 = 1.0 / (t1 - t2)
        #
        d2ye1x = deltaUv2.y() * edge1.x()
        d2ye1y = deltaUv2.y() * edge1.y()
        d2ye1z = deltaUv2.y() * edge1.z()
        #
        d1ye2x = deltaUv1.y() * edge2.x()
        d1ye2y = deltaUv1.y() * edge2.y()
        d1ye2z = deltaUv1.y() * edge2.z()
        #
        tangent = QVector3D()
        tangent.setX(t3 * (d2ye1x - d1ye2x))
        tangent.setY(t3 * (d2ye1y - d1ye2y))
        tangent.setZ(t3 * (d2ye1z - d1ye2z))
        tangent.normalize()
        #
        bitangent = QVector3D()
        bitangent.setX(t3 * (-d2ye1x + d1ye2x))
        bitangent.setY(t3 * (-d2ye1y + d1ye2y))
        bitangent.setZ(t3 * (-d2ye1z + d1ye2z))
        bitangent.normalize()
        #
        return tangent, bitangent

    def makeCoordList(self, tangent1, bitangent1, tangent2, bitangent2):
        "add coords to vertlist"
        t1x = tangent1.x()
        t1y = tangent1.y()
        t1z = tangent1.z()
        t2x = tangent2.x()
        t2y = tangent2.y()
        t2z = tangent2.z()
        bit1x = bitangent1.x()
        bit1y = bitangent1.y()
        bit1z = bitangent1.z()
        bit2x = bitangent2.x()
        bit2y = bitangent2.y()
        bit2z = bitangent2.z()
        coordList = [
            # first triangle
            self.coords["pos"]["tl"].x(),
            self.coords["pos"]["tl"].y(),
            self.coords["pos"]["tl"].z(),
            self.coords["n"].x(),
            self.coords["n"].y(),
            self.coords["n"].z(),
            self.coords["uv"]["tl"].x(),
            self.coords["uv"]["tl"].y(),
            t1x,
            t1y,
            t1z,
            bit1x,
            bit1y,
            bit1z,
            self.coords["pos"]["bl"].x(),
            self.coords["pos"]["bl"].y(),
            self.coords["pos"]["bl"].z(),
            self.coords["n"].x(),
            self.coords["n"].y(),
            self.coords["n"].z(),
            self.coords["uv"]["bl"].x(),
            self.coords["uv"]["bl"].y(),
            t1x,
            t1y,
            t1z,
            bit1x,
            bit1y,
            bit1z,
            self.coords["pos"]["br"].x(),
            self.coords["pos"]["br"].y(),
            self.coords["pos"]["br"].z(),
            self.coords["n"].x(),
            self.coords["n"].y(),
            self.coords["n"].z(),
            self.coords["uv"]["br"].x(),
            self.coords["uv"]["br"].y(),
            t1x,
            t1y,
            t1z,
            bit1x,
            bit1y,
            bit1z,
            # second triangle
            self.coords["pos"]["tl"].x(),
            self.coords["pos"]["tl"].y(),
            self.coords["pos"]["tl"].z(),
            self.coords["n"].x(),
            self.coords["n"].y(),
            self.coords["n"].z(),
            self.coords["uv"]["tl"].x(),
            self.coords["uv"]["tl"].y(),
            t2x,
            t2y,
            t2z,
            bit2x,
            bit2y,
            bit2z,
            self.coords["pos"]["tr"].x(),
            self.coords["pos"]["tr"].y(),
            self.coords["pos"]["tr"].z(),
            self.coords["n"].x(),
            self.coords["n"].y(),
            self.coords["n"].z(),
            self.coords["uv"]["tr"].x(),
            self.coords["uv"]["tr"].y(),
            t2x,
            t2y,
            t2z,
            bit2x,
            bit2y,
            bit2z,
            self.coords["pos"]["br"].x(),
            self.coords["pos"]["br"].y(),
            self.coords["pos"]["br"].z(),
            self.coords["n"].x(),
            self.coords["n"].y(),
            self.coords["n"].z(),
            self.coords["uv"]["br"].x(),
            self.coords["uv"]["br"].y(),
            t2x,
            t2y,
            t2z,
            bit2x,
            bit2y,
            bit2z,
        ]
        return coordList

    def fromCoords2Vertices(self):
        "transform coordinates to vertices"
        edge1 = self.coords["pos"]["bl"] - self.coords["pos"]["tl"]
        edge2 = self.coords["pos"]["br"] - self.coords["pos"]["tl"]
        delta1 = self.coords["uv"]["bl"] - self.coords["uv"]["tl"]
        delta2 = self.coords["uv"]["br"] - self.coords["uv"]["tl"]
        tangent1, bitangent1 = self.computeTangentBiTangent(
            edge1, edge2, delta1, delta2
        )
        edge1 = self.coords["pos"]["br"] - self.coords["pos"]["tl"]
        edge2 = self.coords["pos"]["tr"] - self.coords["pos"]["tl"]
        delta1 = self.coords["uv"]["br"] - self.coords["uv"]["tl"]
        delta2 = self.coords["uv"]["tr"] - self.coords["uv"]["tl"]
        tangent2, bitangent2 = self.computeTangentBiTangent(
            edge1, edge2, delta1, delta2
        )
        clist = self.makeCoordList(tangent1, bitangent1, tangent2, bitangent2)
        self.vertices = np.array(clist, dtype=ctypes.c_float)

    def setObjectShaderUniforms_proc(self):
        "set object shader uniforms"
        projectionMatrix = QMatrix4x4()
        projectionMatrix.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        viewMatrix = self.camera.getViewMatrix()
        model = QMatrix4x4()
        color = self.lamp.diffuse.color
        pos = self.lamp.position
        self.program.setUniformValue("projection", projectionMatrix)
        self.program.setUniformValue("view", viewMatrix)
        self.program.setUniformValue("model", model)
        self.program.setUniformValue("lightPos", self.lamp.position)
        self.program.setUniformValue("viewPos", self.camera.position)
        self.program.setUniformValue("ambientCoeff", self.lamp.ambient.coeffs)
        self.program.setUniformValue("shininess", self.shininess)
        self.program.setUniformValue("lightColor", self.lamp.diffuse.color)


class PtmPerChannelNormalMapGLWidget(PtmNormalMapGLWidget):
    "implement per channel normal map shader with opengl widget"

    def __init__(
        self,
        ptmImage: QImage,
        normalMaps: Tuple[QImage],
        objectShaderName: str,
        lampShaderName="lamp",
        parent=None,
    ):
        super().__init__(
            ptmImage=ptmImage,
            normalMap=normalMaps[0],
            lampShaderName=lampShaderName,
            objectShaderName=objectShaderName,
            parent=parent,
        )
        self.textures = [
            {
                "texture": None,
                "unit": 0,
                "name": "material.diffuseMap",
                "data": ptmImage.mirrored(),
            }
        ]
        for i in range(len(normalMaps)):
            nmap = normalMaps[i].mirrored()
            self.textures.append(
                {
                    "unit": i + 1,
                    "data": nmap,
                    "name": "material.normalMap" + str(i),
                    "texture": None,
                }
            )
        #
        self.vertices = None
        self.fromCoords2Vertices()

    def setObjectShaderUniforms_proc(self):
        "set object shader uniforms"
        raise NotImplemented


class PtmPerChannelNormalMapPhongGLWidget(PtmNormalMapGLWidget):
    "Implement per channel normal map phong shader with opengl widget"

    def __init__(
        self,
        ptmImage: QImage,
        normalMaps: Tuple[QImage],
        objectShaderName="phong",
        lampShaderName="lamp",
        parent=None,
    ):
        super().__init__(
            ptmImage=ptmImage,
            normalMap=normalMaps[0],
            lampShaderName=lampShaderName,
            objectShaderName=objectShaderName,
            parent=parent,
        )
        self.textures = [
            {
                "texture": None,
                "unit": 0,
                "name": "material.diffuseMap",
                "data": ptmImage.mirrored(),
            }
        ]
        for i in range(len(normalMaps)):
            nmap = normalMaps[i].mirrored()
            self.textures.append(
                {
                    "unit": i + 1,
                    "data": nmap,
                    "name": "material.normalMap" + str(i),
                    "texture": None,
                }
            )
        #
        # fmt: off
        self.vertices = np.array(
            [
                # position       # normal       # texture coordinates
                0.7, 0.7, -2.0, 0.0, 0.0, -1.0, 1.0, 1.0,  # top right
                0.7, -0.7, -2.0, 0.0, 0.0, -1.0, 1.0, 0.0,  # bottom right
                -0.7, -0.7, -2.0, 0.0, 0.0, -1.0, 0.0, 0.0,  # bottom left

                -0.7, 0.7, -2.0, 0.0, 0.0, -1.0, 0.0, 1.0,  # top left
                0.7, 0.7, -2.0, 0.0, 0.0, -1.0, 1.0, 1.0,  # top right
                -0.7, -0.7, -2.0, 0.0, 0.0, -1.0, 0.0, 0.0,  # bottom left
            ],
            dtype=ctypes.c_float,
        )
        # fmt: on

    def setObjectShaderUniforms_proc(self):
        "set object shader"
        projectionMatrix = QMatrix4x4()
        projectionMatrix.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        viewMatrix = self.camera.getViewMatrix()
        model = QMatrix4x4()
        color = self.lamp.diffuse.color
        pos = self.lamp.position
        self.program.setUniformValue("projection", projectionMatrix)
        self.program.setUniformValue("view", viewMatrix)
        self.program.setUniformValue("model", model)
        self.program.setUniformValue("viewerPosition", self.camera.position)
        self.program.setUniformValue("light.position", pos)
        self.program.setUniformValue("light.direction", self.lamp.front)
        self.program.setUniformValue("light.color", color)
        self.program.setUniformValue("light.attenuation", self.lamp.attenuation)
        self.program.setUniformValue("light.cutOff", self.lamp.cutOff)
        self.program.setUniformValue("light.outerCutOff", self.lamp.outerCutOff)
        self.program.setUniformValue("material.shininess", self.shininess)
        self.program.setUniformValue("ambientCoeff", self.lamp.ambient.coeffs)


class PtmPerChannelNormalMapDirGLWidget(PtmPerChannelNormalMapGLWidget):
    "directional light normal mapping"

    def __init__(
        self,
        ptmImage: QImage,
        normalMaps: Tuple[QImage],
        objectShaderName="quadDir",
        lampShaderName="lamp",
        parent=None,
    ):
        super().__init__(
            ptmImage=ptmImage,
            normalMaps=normalMaps,
            lampShaderName=lampShaderName,
            objectShaderName=objectShaderName,
            parent=parent,
        )

    def setObjectShaderUniforms_proc(self):
        "set object shader"
        projectionMatrix = QMatrix4x4()
        projectionMatrix.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        viewMatrix = self.camera.getViewMatrix()
        model = QMatrix4x4()
        color = self.lamp.diffuse.color
        pos = self.lamp.position
        self.program.setUniformValue("projection", projectionMatrix)
        self.program.setUniformValue("view", viewMatrix)
        self.program.setUniformValue("model", model)
        self.program.setUniformValue("viewPos", self.camera.position)
        self.program.setUniformValue("light.direction", self.lamp.front)
        self.program.setUniformValue("light.color", color)
        self.program.setUniformValue("material.shininess", self.shininess)
        self.program.setUniformValue("ambient", self.lamp.ambient.coeffs)


class PtmPerChannelNormalMapPointGLWidget(PtmPerChannelNormalMapGLWidget):
    "point light normal mapping"

    def __init__(
        self,
        ptmImage: QImage,
        normalMaps: Tuple[QImage],
        objectShaderName="quadPoint",
        lampShaderName="lamp",
        parent=None,
    ):
        super().__init__(
            ptmImage=ptmImage,
            normalMaps=normalMaps,
            lampShaderName=lampShaderName,
            objectShaderName=objectShaderName,
            parent=parent,
        )

    def setObjectShaderUniforms_proc(self):
        "set object shader"
        projectionMatrix = QMatrix4x4()
        projectionMatrix.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        viewMatrix = self.camera.getViewMatrix()
        model = QMatrix4x4()
        color = self.lamp.diffuse.color
        pos = self.lamp.position
        self.program.setUniformValue("projection", projectionMatrix)
        self.program.setUniformValue("view", viewMatrix)
        self.program.setUniformValue("model", model)
        self.program.setUniformValue("viewPos", self.camera.position)
        self.program.setUniformValue("light.position", pos)
        self.program.setUniformValue("light.color", color)
        self.program.setUniformValue("light.attenuation", self.lamp.attenuation)

        self.program.setUniformValue("material.shininess", self.shininess)
        self.program.setUniformValue("ambient", self.lamp.ambient.coeffs)


class PtmPerChannelNormalMapSpotGLWidget(PtmPerChannelNormalMapGLWidget):
    "directional light normal mapping"

    def __init__(
        self,
        ptmImage: QImage,
        normalMaps: Tuple[QImage],
        objectShaderName="quadSpot",
        lampShaderName="lamp",
        parent=None,
    ):
        super().__init__(
            ptmImage=ptmImage,
            normalMaps=normalMaps,
            lampShaderName=lampShaderName,
            objectShaderName=objectShaderName,
            parent=parent,
        )

    def setObjectShaderUniforms_proc(self):
        "set object shader"
        projectionMatrix = QMatrix4x4()
        projectionMatrix.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        viewMatrix = self.camera.getViewMatrix()
        model = QMatrix4x4()
        color = self.lamp.diffuse.color
        pos = self.lamp.position
        self.program.setUniformValue("projection", projectionMatrix)
        self.program.setUniformValue("view", viewMatrix)
        self.program.setUniformValue("model", model)
        self.program.setUniformValue("viewPos", self.camera.position)
        self.program.setUniformValue("light.position", pos)
        self.program.setUniformValue("light.direction", self.lamp.front)
        self.program.setUniformValue("light.color", color)
        self.program.setUniformValue("light.attenuation", self.lamp.attenuation)
        self.program.setUniformValue("light.cutOff", self.lamp.cutOff)
        self.program.setUniformValue("light.outerCutOff", self.lamp.outerCutOff)

        self.program.setUniformValue("material.shininess", self.shininess)
        self.program.setUniformValue("ambient", self.lamp.ambient.coeffs)


class PtmCoefficientShader(AbstractPointLightPtmGLWidget):
    "Use directly ptm coefficients in shader"

    def __init__(
        self,
        coeffs: np.ndarray,
        vertexNb: int,
        lampShaderName="lamp",
        objectShaderName="rgbcoeff",
        parent=None,
    ):
        super().__init__(parent=parent)
        self.vertices = coeffs
        self.isBlinn = False
        self.objectShaderName = objectShaderName
        self.lampShaderName = lampShaderName
        self.vertexNb = vertexNb

    def programShader_init_uniforms(self):
        "set uniforms during initilization"
        return

    def lampShader_init_uniforms(self):
        "set uniforms during initilization of lamp shader"
        return

    def setLampShaderUniforms_proc(self):
        "Set lamp shader uniforms"
        projectionMatrix = QMatrix4x4()
        projectionMatrix.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        viewMatrix = self.camera.getViewMatrix()
        lampModel = self.lamp.get_model_matrix()
        self.lampProgram.setUniformValue("projection", projectionMatrix)
        self.lampProgram.setUniformValue("view", viewMatrix)
        self.lampProgram.setUniformValue("model", lampModel)

    def setObjectShaderUniforms_proc(self):
        "set uniforms to object shader"
        projectionMatrix = QMatrix4x4()
        projectionMatrix.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        model = QMatrix4x4()
        viewMatrix = self.camera.getViewMatrix()
        self.program.setUniformValue("projection", projectionMatrix)
        self.program.setUniformValue("view", viewMatrix)
        self.program.setUniformValue("model", model)
        self.program.setUniformValue("viewPos", self.camera.position)
        self.program.setUniformValue("lightPos", self.lamp.position)
        self.program.setUniformValue("lightColor", self.lamp.diffuse.color)
        self.program.setUniformValue("attc", self.lamp.attenuation)
        self.program.setUniformValue("blinn", self.isBlinn)
        self.program.setUniformValue("diffuseCoeffs", self.lamp.diffuse.coeffs)
        self.program.setUniformValue("specularCoeffs", self.lamp.specular.coeffs)
        self.program.setUniformValue("ambientCoeffs", self.lamp.ambient.coeffs)

    def createTextures(self):
        "no texture in this shader"
        return

    def initializeGL(self):
        # create context and make it current
        self.context.create()
        self.context.aboutToBeDestroyed.connect(self.cleanUpGL)

        # initialize functions
        funcs = self.context.functions()
        funcs.initializeOpenGLFunctions()
        funcs.glClearColor(0.0, 0.4, 0.4, 0)
        funcs.glEnable(pygl.GL_DEPTH_TEST)

        # vars to be used in initialization
        floatSize = ctypes.sizeof(ctypes.c_float)
        # create lamp shader
        self.lampProgram = QOpenGLShaderProgram(self.context)
        self.lampShader_init(self.lampShaderName)
        self.lampProgram.bind()
        self.lampShader_init_uniforms()

        # create object shader
        self.program = QOpenGLShaderProgram(self.context)
        self.programShader_init(self.objectShaderName)
        self.program.bind()
        self.programShader_init_uniforms()

        # lamp: vbo
        self.lampVbo.create()
        self.lampVbo.bind()
        self.lampVbo.allocate(
            self.lampVertices.tobytes(), self.lampVertices.size * floatSize
        )

        # lamp: vao
        self.lampVao.create()
        self.lampVao.bind()
        stride = self.attrLoc[self.lampShaderName]["stride"] * floatSize
        # position
        for attrName, adict in self.attrLoc[self.lampShaderName].items():
            if attrName != "stride":
                layout = adict["layout"]
                size = adict["size"]
                offset = adict["offset"] * floatSize
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
        self.lampVao.release()
        self.lampVbo.release()

        # object: vbo, vao
        self.vbo.create()
        self.vbo.bind()
        self.vbo.allocate(self.vertices.tobytes(), self.vertices.size * floatSize)
        self.vbo.allocate(self.vertices.tobytes(), self.vertices.size * floatSize)
        #
        self.vao.create()
        self.vao.bind()
        #
        stride = self.attrLoc[self.objectShaderName]["stride"] * floatSize
        print("stride val with float:", stride)
        print("stride val:", self.attrLoc[self.objectShaderName]["stride"])
        for attrName, adict in self.attrLoc[self.objectShaderName].items():
            if attrName != "stride":
                layout = adict["layout"]
                size = adict["size"]
                offset = adict["offset"] * floatSize
                funcs.glEnableVertexAttribArray(layout)
                funcs.glVertexAttribPointer(
                    layout,
                    size,
                    int(pygl.GL_FLOAT),
                    int(pygl.GL_FALSE),
                    stride,
                    VoidPtr(offset),
                )
        # end object: vbo, vao
        self.vbo.release()
        self.vao.release()
        # create texture
        self.createTextures()

    def paintGL(self):
        "paint gl drawing loop"
        funcs = self.context.functions()
        funcs.glClear(pygl.GL_COLOR_BUFFER_BIT | pygl.GL_DEPTH_BUFFER_BIT)
        #
        # render object
        self.vao.bind()
        self.program.bind()
        self.setObjectShaderUniforms_proc()
        # self.bindTextures_proc()
        funcs.glDrawArrays(pygl.GL_POINTS, 0, self.vertexNb)

        # render lamp
        self.lampVao.bind()
        self.lampProgram.bind()
        self.setLampShaderUniforms_proc()
        funcs.glDrawArrays(pygl.GL_TRIANGLES, 0, 36)
