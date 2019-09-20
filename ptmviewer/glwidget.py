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
from PySide2.QtGui import QOpenGLContext
from PySide2.QtGui import QMatrix4x4
from PySide2.QtGui import QVector4D
from PySide2.QtGui import QColor
from PySide2.QtGui import QImage


from PySide2.QtWidgets import QApplication
from PySide2.QtWidgets import QMessageBox
from PySide2.QtWidgets import QOpenGLWidget

from PySide2.QtCore import QCoreApplication


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
    messageBox.setDetailedText("Run:\npip install PyOpenGL PyOpenGL_accelerate")
    messageBox.exec_()
    sys.exit(1)

from PySide2.QtWidgets import QOpenGLWidget


class PtmGLWidget(QOpenGLWidget):
    "OpenGL widget"

    def __init__(self, surfaceNormals: QImage, texture: QImage, parent=None):
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
            "aNormal": {"stride": 3, "offset": 3, "layout": 1},
            "aTexCoord": {"stride": 2, "offset": 5, "layout": 2},
            "aTangent": {"stride": 3, "offset": 8, "layout": 3},
            "aBiTangent": {"stride": 3, "offset": 11, "layout": 4},
        }

        # opengl data

        self.context = QOpenGLContext()
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)

        self.lampVao = QOpenGLVertexArrayObject()
        self.lampVbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.program = QOpenGLShaderProgram()
        self.lampProgram = QOpenGLShaderProgram()
        self.diffuseMap = None  # texture
        self.specularMap = None  # texture
        self.texUnit1 = 0
        self.texUnit2 = 1
        # texture
        self.textureImage = texture.mirrored()
        self.textureNormals = surfaceNormals.mirrored()
        imrect = self.textureImage.rect()
        imwidth = imrect.width()
        imheight = imrect.height()
        imcenter = imrect.center()
        imRightX = imcenter.x()
        imTopY = imcenter.y()
        imLeftX = imRightX - imwidth
        imBottomY = imTopY - imheight

        # vertices
        self.textureViewerTLeft = QVector3D(imLeftX, imTopY, 0.0)
        self.textureViewerBLeft = QVector3D(imLeftX, imBottomY, 0.0)
        self.textureViewerBRight = QVector3D(imRightX, imBottomY, 0.0)
        self.textureViewerTRight = QVector3D(imRightX, imTopY, 0.0)
        self.firstTriangle = (
            self.textureViewerTLeft,
            self.textureViewerBLeft,
            self.textureViewerBRight,
        )
        self.secondTriangle = (
            self.textureViewerTLeft,
            self.textureViewerTRight,
            self.textureViewerBRight,
        )

        self.textureCoordCorner1 = QVector2D(0.0, 1.0)
        self.textureCoordCorner2 = QVector2D(0.0, 0.0)
        self.textureCoordCorner3 = QVector2D(1.0, 1.0)
        self.textureCoordCorner4 = QVector2D(1.0, 1.0)

        self.firstTexTriangle = (
            self.textureCoordCorner1,
            self.textureCoordCorner2,
            self.textureCoordCorner3,
        )
        self.secondTexTriangle = (
            self.textureCoordCorner1,
            self.textureCoordCorner3,
            self.textureCoordCorner4,
        )

        self.viewerNormal = QVector3D(0.0, 0.0, 1.0)

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

    def diffuseMap_proc(self):
        self.diffuseMap = QOpenGLTexture(QOpenGLTexture.Target2D)
        self.diffuseMap.create()
        self.diffuseMap.bind(self.texUnit1)
        self.diffuseMap.setData(self.textureImage)
        self.diffuseMap.setMinMagFilters(QOpenGLTexture.Nearest, QOpenGLTexture.Nearest)
        self.diffuseMap.setWrapMode(QOpenGLTexture.DirectionS, QOpenGLTexture.Repeat)
        self.diffuseMap.setWrapMode(QOpenGLTexture.DirectionT, QOpenGLTexture.Repeat)

    def normalMap_proc(self):
        self.normalMap = QOpenGLTexture(QOpenGLTexture.Target2D)
        self.normalMap.create()
        self.normalMap.bind(self.texUnit2)
        self.normalMap.setData(self.textureNormals)
        self.normalMap.setMinMagFilters(
            QOpenGLTexture.LinearMipMapLinear, QOpenGLTexture.Linear
        )
        self.normalMap.setWrapMode(QOpenGLTexture.DirectionS, QOpenGLTexture.Repeat)
        self.normalMap.setWrapMode(QOpenGLTexture.DirectionT, QOpenGLTexture.Repeat)

    def computeTangentBiTangent(
        self,
        edge1: QVector3D,
        edge2: QVector3D,
        deltaUv1: QVector3D,
        deltaUv2: QVector3D,
    ):
        "Compute tangent and bi tangent vectors"
        d12xy = deltaUv1.x() * deltaUv2.y()
        d21xy = deltaUv2.x() * deltaUv1.y()
        coeff = d12xy - d21xy
        coeff = 1.0 / coeff
        #
        tangent = QVector3D()
        dY2 = deltaUv2.y()
        dY1 = deltaUv1.y()
        dX1 = deltaUv1.x()
        dX2 = deltaUv2.x()
        # x
        ex1 = edge1.x()
        ex2 = edge2.x()
        # y
        ey1 = edge1.y()
        ey2 = edge2.y()
        # z
        ez1 = edge1.z()
        ez2 = edge2.z()
        # tangent
        tangent.setX(coeff * ((dY2 * ex1) - (dY1 * ex2)))
        tangent.setY(coeff * ((dY2 * ey1) - (dY1 * ey2)))
        tangent.setZ(coeff * ((dY2 * ez1) - (dY1 * ez2)))
        tangent.normalize()
        # bi tangent
        bitangent = QVector3D()
        bitangent.setX(coeff * ((-dX2 * ex1) + (dX1 * ex2)))
        bitangent.setY(coeff * ((-dX2 * ey1) + (dX1 * ey2)))
        bitangent.setZ(coeff * ((-dX2 * ez1) + (dX1 * ez2)))
        bitangent.normalize()
        #
        return tangent, bitangent

    def addQuadVertice(self, vertices: list, values: list):
        "Add values to vertices"
        for val in values:
            vertices.append(val)
        #
        return vertices

    def makeQuadRow(self, rowVals: dict):
        "Make quad row"
        texCoord = rowVals["texCoord"]
        pos = rowVals["position"]
        normal = rowVals["normal"]
        tangent = rowVals["tangent"]
        bitangent = rowVals["bitangent"]
        return (
            pos.x(),
            pos.y(),
            pos.z(),
            normal.x(),
            normal.y(),
            normal.z(),
            texCoord.x(),
            texCoord.y(),
            tangent.x(),
            tangent.y(),
            tangent.z(),
            bitangent.x(),
            bitangent.y(),
            bitangent.z(),
        )

    def addQuadTriangle(self, vertices: list, textureTriangle, surfaceTriangle):
        "Add triangle to quad vertices"
        edge1 = surfaceTriangle[1] - surfaceTriangle[0]
        edge2 = surfaceTriangle[2] - surfaceTriangle[0]
        deltaUv1 = textureTriangle[1] - textureTriangle[0]
        deltaUv2 = textureTriangle[2] - textureTriangle[0]
        tangent, bitangent = self.computeTangentBiTangent(
            edge1, edge2, deltaUv1, deltaUv2
        )
        for cornerNb in range(3):
            self.addQuadVertice(
                vertices=vertices,
                values=self.makeQuadRow(
                    rowVals={
                        "texCoord": textureTriangle[cornerNb],
                        "position": surfaceTriangle[cornerNb],
                        "normal": self.viewerNormal,
                        "tangent": tangent,
                        "bitangent": bitangent,
                    }
                ),
            )
        #
        return vertices

    def renderViewer(self):
        "Render viewer object with texture object"
        quadVertices = []
        quadVertices = self.addQuadTriangle(
            vertices=quadVertices,
            textureTriangle=self.firstTexTriangle,
            surfaceTriangle=self.firstTriangle,
        )
        quadVertices = self.addQuadTriangle(
            vertices=quadVertices,
            textureTriangle=self.secondTexTriangle,
            surfaceTriangle=self.secondTriangle,
        )
        quadVertices = np.array(quadVertices, dtype=ctypes.c_float)
        floatSize = ctypes.sizeof(ctypes.c_float)
        self.vbo.create()
        self.vbo.bind()
        self.vbo.allocate(quadVertices.tobytes(), floatSize * quadVertices.size)
        self.vao.create()
        self.enableVaoAttribs()
        self.vao.bind()
        funcs.glDrawArrays(pygl.GL_TRIANGLES, 0, 6)
        self.vao.release()

    def enableVaoAttribs(self):
        "enable attributes of vao"
        funcs = self.context.functions()
        rowsize = 0
        for key, valdi in self.attrLoc.items():
            rowsize += valdi["stride"]
        #
        ks = ["layout", "stride", "offset"]
        for key, valdi in self.attrLoc.items():
            funcs.glEnableVertexAttribArray(valdi[ks[0]])
            funcs.glVertexAttribPointer(
                valdi[ks[0]],
                valdi[ks[1]],
                int(pygl.GL_FLOAT),
                int(pygl.GL_FALSE),
                rowsize,
                VoidPtr(valdi[ks[2]] * floatSize),
            )

    def setObjectUniforms_proc(self):
        "Set object uniforms program"
        # vertex shader
        # set projection
        projection = QMatrix4x4()
        projection.perspective(
            self.camera.zoom, self.width() / self.height(), 0.2, 100.0
        )
        self.program.setUniformValue("projection", projection)
        view = self.camera.getViewMatrix()
        self.program.setUniformValue("view", view)
        model = QMatrix4x4()
        self.program.setUniformValue("model", model)
        viewPos = self.camera.position
        self.program.setUniformValue("viewPos", viewPos)
        lightPos = self.lamp.position
        self.program.setUniformValue("lightPos", lightPos)
        # end vertex shader uniforms
        # fragment shader
        self.program.setUniformValue("lightColor", self.lamp.color)
        self.program.setUniformValue("ambientCoeff", self.ambientCoeff)
        self.program.setUniformValue("shininess", self.shininess)
        # end fragment shader

    def objectShader_init(self):
        "Object shader initialization"
        shader = "quad"
        vshader = self.loadVertexShader(shader, fromFile=False)
        fshader = self.loadFragmentShader(shader, fromFile=False)
        self.program.addShader(vshader)
        self.program.addShader(fshader)
        linked = self.program.link()
        print("quad shader linked: ", linked)

    def lampShader_init(self):
        "Lamp shader initialization"
        shader = "lamp"
        vshader = self.loadVertexShader(shader, fromFile=False)
        fshader = self.loadFragmentShader(shader, fromFile=False)
        self.lampProgram.addShader(vshader)
        self.lampProgram.addShader(fshader)
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
        funcs.glEnable(pygl.GL_TEXTURE_2D)

        # load shaders: lamp shader, object shader
        # initialize shaders
        self.program = QOpenGLShaderProgram(self.context)
        self.objectShader_init()
        print("object init done")
        isb = self.program.bind()
        print("object shader bound: ", isb)

        # set uniforms to shaders
        self.program.setUniformValue("diffuseMap", self.texUnit1)
        self.program.setUniformValue("normalMap", self.texUnit2)

        self.lampProgram = QOpenGLShaderProgram(self.context)
        self.lampShader_init()
        print("lamp init done")
        # set vao and vbo for shaders
        isb = self.lampProgram.bind()
        print("lamp shader bound: ", isb)

        floatSize = ctypes.sizeof(ctypes.c_float)
        # lamp vbo
        self.lampVbo.create()
        self.lampVao.create()
        #
        self.lampVbo.bind()
        self.lampVbo.allocate(
            self.lampVertices.tobytes(), floatSize * self.lampVertices.size
        )
        self.lampVao.bind()
        funcs.glEnableVertexAttribArray(self.attrLoc["aPos"]["layout"])
        rowsize = self.attrLoc["aPos"]["stride"] * floatSize
        print("attribute info: ", self.attrLoc)
        print("row size info: ", rowsize)
        funcs.glVertexAttribPointer(
            self.attrLoc["aPos"]["layout"],
            self.attrLoc["aPos"]["stride"],
            int(pygl.GL_FLOAT),
            int(pygl.GL_FALSE),
            rowsize,
            VoidPtr(self.attrLoc["aPos"]["offset"] * floatSize),
        )
        # textures
        self.diffuseMap_proc()
        self.normalMap_proc()

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
        self.diffuseMap.bind()
        self.normalMap.bind()
        # render object
        self.renderViewer()

        # bind shader: lamp
        self.lampProgram.bind()
        # set uniforms to shader
        self.setLampShaderUniforms()
        # render lamp
        self.lampVao.bind()
        funcs.glDrawArrays(pygl.GL_TRIANGLES, 0, 36)
