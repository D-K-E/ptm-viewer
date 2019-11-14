# author: Kaan Eraslan
# camera

import numpy as np
import math
from ptmviewer.utils.utils import computeLookAtPure
from ptmviewer.utils.utils import normalize_tuple
from ptmviewer.utils.utils import crossProduct
from ptmviewer.utils.utils import scalar2vecMult
from ptmviewer.utils.utils import vec2vecAdd
from ptmviewer.utils.utils import vec2vecSubs
from ptmviewer.utils.utils import move3dObjPure
from ptmviewer.utils.utils import move3dObjQt
from PySide2.QtGui import QVector3D
from PySide2.QtGui import QMatrix4x4
from PySide2.QtGui import QVector4D
from ptmviewer.utils.obj3d import PureRigid3dObject
from ptmviewer.utils.obj3d import QtRigid3dObject


class PureCamera(PureRigid3dObject):
    "A camera that is in pure python for 3d movement"

    def __init__(self):
        ""
        super().__init__()

        # movement speed, sensitivity, moves, zoom
        self.zoom = 45.0

        # update camera vectors
        self.updateVectors()

    def lookAround(self, xoffset: float, yoffset: float, pitchBound: bool):
        "Look around with camera"
        xoffset *= self.movementSensitivity
        yoffset *= self.movementSensitivity
        self.yaw += xoffset
        self.pitch += yoffset

        if pitchBound:
            if self.pitch > 90.0:
                self.pitch = 90.0
            elif self.pitch < -90.0:
                self.pitch = -90.0
        #
        self.updateVectors()

    def zoomInOut(self, yoffset: float, zoomBound=45.0):
        "Zoom with camera"
        if self.zoom >= 1.0 and self.zoom <= zoomBound:
            self.zoom -= yoffset
        elif self.zoom <= 1.0:
            self.zoom = 1.0
        elif self.zoom >= zoomBound:
            self.zoom = zoomBound

    def setCameraWithVectors(
        self,
        position: dict,
        up: dict,
        front: dict,
        yaw: float,
        pitch: float,
        zoom: float,
        speed: float,
        sensitivity: float,
    ):
        "Set camera"
        self.check_coordinate_proc(position)
        self.check_coordinate_proc(up)
        self.check_coordinate_proc(front)
        self.position = position
        self.worldUp = up
        self.pitch = pitch
        self.yaw = yaw
        self.movementSpeed = speed
        self.movementSensitivity = sensitivity
        self.front = front
        self.zoom = zoom
        self.updateVectors()

    def setCameraWithFloatVals(
        self,
        posx: float,
        posy: float,
        posz: float,
        upx: float,
        upy: float,
        upz: float,
        yaw: float,
        pitch: float,
        speed: float,
        sensitivity: float,
        zoom: float,
        front: dict,
    ):
        "Set camera floats"
        self.check_coordinate_proc(front)
        self.position = {"x": posx, "y": posy, "z": posz}
        self.worldUp = {"x": upx, "y": upy, "z": upz}
        self.yaw = yaw
        self.pitch = pitch
        self.movementSpeed = speed
        self.movementSensitivity = sensitivity
        self.zoom = zoom
        self.front = front
        self.updateVectors()

    def getViewMatrix(self):
        "Obtain view matrix for camera"
        pos = (self.position["x"], self.position["y"], self.position["z"])
        front = (self.front["x"], self.front["y"], self.front["z"])
        wup = (self.worldUp["x"], self.worldUp["y"], self.worldUp["z"])
        return computeLookAtPure(pos=pos, target=vec2vecAdd(pos, front), worldUp=wup)

    def __str__(self):
        "string representation"
        mess = "Camera: position {0},\n yaw: {1},\n pitch: {2},\n world up:{3}"
        mes = mess.format(
            str(self.position), str(self.yaw), str(self.pitch), str(self.worldUp)
        )
        return mes


class QtCamera(QtRigid3dObject):
    "An abstract camera for 3d movement in world"

    def __init__(self):
        ""
        super().__init__()
        # Camera attributes
        self.front = QVector3D(0.0, 0.0, -0.5)
        self.worldUp = QVector3D(0.0, 1.0, 0.0)

        # Euler Angles for rotation
        self.yaw = -90.0
        self.pitch = 0.0

        # camera options
        self.movementSpeed = 2.5
        self.movementSensitivity = 0.00001
        self.zoom = 45.0

    def updateVectors(self):
        "Update the camera vectors and compute a new front"
        yawRadian = np.radians(self.yaw)
        yawCos = np.cos(yawRadian)
        pitchRadian = np.radians(self.pitch)
        pitchCos = np.cos(pitchRadian)
        frontX = yawCos * pitchCos
        frontY = np.sin(pitchRadian)
        frontZ = np.sin(yawRadian) * pitchCos
        self.front = QVector3D(frontX, frontY, frontZ)
        self.front.normalize()
        self.right = QVector3D.crossProduct(self.front, self.worldUp)
        self.right.normalize()
        self.up = QVector3D.crossProduct(self.right, self.front)
        self.up.normalize()

    def lookAround(self, xoffset: float, yoffset: float, pitchBound: bool):
        "Look around with camera"
        xoffset *= self.movementSensitivity
        yoffset *= self.movementSensitivity
        self.yaw += xoffset
        self.pitch += yoffset

        if pitchBound:
            if self.pitch > 89.9:
                self.pitch = 89.9
            elif self.pitch < -89.9:
                self.pitch = -89.9
        #
        self.updateCameraVectors()

    def zoomInOut(self, yoffset: float, zoomBound=45.0):
        "Zoom with camera"
        if self.zoom >= 1.0 and self.zoom <= zoomBound:
            self.zoom -= yoffset
        elif self.zoom <= 1.0:
            self.zoom = 1.0
        elif self.zoom >= zoomBound:
            self.zoom = zoomBound

    def getViewMatrix(self):
        "Obtain view matrix for camera"
        view = QMatrix4x4()
        view.lookAt(self.position, self.position + self.front, self.up)
        return view

    def setCameraWithVectors(
        self,
        position=QVector3D(0.0, 0.0, 0.0),
        worldUp=QVector3D(0.0, 1.0, 0.0),
        yaw=-90.0,
        pitch=0.0,
        zoom=45.0,
        speed=2.5,
        sensitivity=0.00001,
    ):
        "Set camera"
        self.check_coordinate_proc(position)
        self.check_coordinate_proc(worldUp)
        self.position = position
        self.worldUp = worldUp
        self.pitch = pitch
        self.yaw = yaw
        self.movementSpeed = speed
        self.movementSensitivity = sensitivity
        self.zoom = zoom
        self.updateVectors()

    def setWorldUp(self, worldUp: QVector3D):
        "Set new world up"
        self.check_coordinate_proc(worldUp)
        self.worldUp = worldUp
        self.updateVectors()

    def setPitch(self, pitch: float):
        "Set new pitch and set other stuff"
        self.pitch = pitch
        self.updateVectors()

    def setPosition(self, position: QVector3D):
        "Set camera position and compute other with respect to new position"
        self.check_coordinate_proc(position)
        self.position = position
        self.updateVectors()

    def setYaw(self, yaw: float):
        "Set new yaw and compute other stuff"
        self.yaw = yaw
        self.updateVectors()

    def setCameraWithFloatVals(
        self,
        posx=0.0,
        posy=0.0,
        posz=0.0,
        upx=0.0,
        upy=1.0,
        upz=0.0,
        yaw=-90.0,
        pitch=0.0,
        zoom=45.0,
        speed=2.5,
        sensitivity=0.00001,
    ):
        "Set camera floats"
        self.position = QVector3D(posx, posy, posz)
        self.worldUp = QVector3D(upx, upy, upz)
        self.yaw = yaw
        self.pitch = pitch
        self.movementSpeed = speed
        self.movementSensitivity = sensitivity
        self.zoom = zoom
        self.updateVectors()

    def __str__(self):
        "string representation"
        mess = "Camera: position {0}, yaw: {1}, pitch: {2}, world up:{3}"
        mes = mess.format(
            str(self.position), str(self.yaw), str(self.pitch), str(self.worldUp)
        )
        return mes


class FPSCameraQt(QtCamera):
    "FPS Camera based on qtcamera"

    def __init__(self):
        super().__init__()

    def move(self, direction: str, deltaTime: float):
        "Move camera in single axis"
        self.position = self.move2pos(direction, deltaTime)
        self.position.setY(0.0)  # y val == 0
