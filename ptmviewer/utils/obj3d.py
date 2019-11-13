"""
A module for objects expressed 3d world
"""

import math
from ptmviewer.utils.utils import mat2matDot
from ptmviewer.utils.utils import normalize_tuple
from ptmviewer.utils.utils import vec2vecAdd
from ptmviewer.utils.utils import vec2vecSubs
from ptmviewer.utils.utils import scalar2vecMult
from ptmviewer.utils.utils import crossProduct
from ptmviewer.utils.utils import move3dObjPure
from typing import Tuple, List, Dict
from PySide2.QtGui import QVector3D
from PySide2.QtGui import QMatrix4x4
from PySide2.QtGui import QVector4D


class PureRigid3dObject:
    "Basic rigid body in 3d world"

    def __init__(self):
        self.position = (0.0, 0.0, 0.0)  # object at the center of the world
        self.front = None
        self.worldUp = (0.0, 1.0, 0.0)
        # up with respect to center of the world
        self.up = None  # with respect to the objects current position
        self.right = None  # with respect to the objects current position
        #
        # Euler angles for rotation
        self.yaw = -90.0  # rotation over z axis
        self.pitch = 0.0  # rotation over y axis
        self.roll = 0.0  # rotation over x axis

        # movement
        self.movementSensitivity = 0.001
        self.movementSpeed = 2.5
        self.availableMoves = ["+z", "-z", "+x", "-x", "+y", "-y"]

    @property
    def z_axis_rotation_matrix(self):
        "rotation matrix over z axis"
        return [
            [math.cos(self.yaw), -math.sin(self.yaw), 0, 0],
            [math.sin(self.yaw), math.cos(self.yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]

    @property
    def y_axis_rotation_matrix(self):
        "rotation matrix over y axis"
        return [
            [math.cos(self.pitch), 0, math.sin(self.pitch), 0],
            [0, 1, 0, 0],
            [-math.sin(self.pitch), 0, math.cos(self.pitch), 0],
            [0, 0, 0, 1],
        ]

    @property
    def x_axis_rotation_matrix(self):
        "rotation matrix over x axis"
        return [
            [1, 0, 0, 0],
            [0, math.cos(self.roll), -math.sin(self.roll), 0],
            [0, math.sin(self.roll), math.cos(self.roll), 0],
            [0, 0, 0, 1],
        ]

    @property
    def rotation_matrix(self):
        "rotation matrix"
        mat2 = mat2matDot(
            mat1=self.y_axis_rotation_matrix, mat2=self.x_axis_rotation_matrix
        )
        return mat2matDot(mat1=self.z_axis_rotation_matrix, mat2=mat2)

    def updateVectors(self):
        "update front, up, right vectors"
        yawRadian = math.radians(self.yaw)
        yawCos = math.cos(yawRadian)
        pitchRadian = math.radians(self.pitch)
        pitchCos = math.cos(pitchRadian)
        frontX = yawCos * pitchCos
        frontY = math.sin(pitchRadian)
        frontZ = math.sin(yawRadian) * pitchCos
        self.front = (frontX, frontY, frontZ)
        self.front = normalize_tuple(self.front)
        self.right = crossProduct(self.front, self.worldUp)
        self.right = normalize_tuple(self.right)
        self.up = crossProduct(self.right, self.front)
        self.up = normalize_tuple(self.up)

    def move2pos(self, direction: str, deltaTime: float):
        "move object to given direction"
        velocity = self.movementSpeed * deltaTime
        direction = direction.lower()
        if direction not in self.availableMoves:
            raise ValueError(
                "Unknown direction {0}, available moves are {1}".format(
                    direction, self.availableMoves
                )
            )
        if direction == "+z":
            multip = scalar2vecMult(self.front, velocity)
            positionVector = vec2vecAdd(self.position, multip)
        elif direction == "-z":
            multip = scalar2vecMult(self.front, velocity)
            positionVector = vec2vecSubs(self.position, multip)
        elif direction == "+x":
            multip = scalar2vecMult(self.right, velocity)
            positionVector = vec2vecAdd(self.position, multip)
        elif direction == "-x":
            multip = scalar2vecMult(self.right, velocity)
            positionVector = vec2vecSubs(self.position, multip)
        elif direction == "+y":
            multip = scalar2vecMult(self.up, velocity)
            positionVector = vec2vecAdd(self.position, multip)
        elif direction == "-y":
            multip = scalar2vecMult(self.up, velocity)
            positionVector = vec2vecSubs(self.position, multip)

        return positionVector

    def move(self, direction: str, deltaTime: float):
        "move object to its new position"
        self.position = self.move2pos(direction, deltaTime)
        self.updateVectors()

    def setYaw(self, yaw: float):
        "set yaw value"
        self.yaw = yaw
        self.updateVectors()

    def setPitch(self, pitch: float):
        "set pitch value"
        self.pitch = pitch
        self.updateVectors()

    def setRoll(self, roll: float):
        "set roll value"
        self.roll = roll
        self.updateVectors()

    def setWorldUp(self, wup: Tuple[float, float, float]):
        "world up"
        self.worldUp = wup
        self.updateVectors()

    def setPosition(self, pos: Tuple[float, float, float]):
        "set position"
        self.position = pos
        self.updateVectors()


class QtRigid3dObject(PureRigid3dObject):
    "Rigid 3d object with qt constructs"

    def __init__(self):
        super().__init__()
        self.position = QVector3D(0.0, 0.0, 0.0)
        self.front = None
        self.worldUp = QVector3D(0.0, 1.0, 0.0)
        self.up = None
        self.right = None

    # overriding property
    @property
    def x_axis_rotation_matrix(self):
        "overriding base class property"
        return QMatrix4x4(
            [
                QVector4D(1, 0, 0, 0),
                QVector4D(0, math.cos(self.roll), -math.sin(self.roll), 0),
                QVector4D(0, math.sin(self.roll), math.cos(self.roll), 0),
                QVector4D(0, 0, 0, 1),
            ]
        )

    @property
    def y_axis_rotation_matrix(self):
        "rotation matrix over y axis"
        return QMatrix4x4(
            [
                QVector4D(math.cos(self.pitch), 0, math.sin(self.pitch), 0),
                QVector4D(0, 1, 0, 0),
                QVector4D(-math.sin(self.pitch), 0, math.cos(self.pitch), 0),
                QVector4D(0, 0, 0, 1),
            ]
        )

    @property
    def z_axis_rotation_matrix(self):
        return QMatrix4x4(
            [
                QVector4D(math.cos(self.yaw), -math.sin(self.yaw), 0, 0),
                QVector4D(math.sin(self.yaw), math.cos(self.yaw), 0, 0),
                QVector4D(0, 0, 1, 0),
                QVector4D(0, 0, 0, 1),
            ]
        )

    @property
    def rotation_matrix(self):
        "rotation matrix"
        mat2 = self.y_axis_rotation_matrix * self.x_axis_rotation_matrix
        return self.z_axis_rotation_matrix * mat2

    def updateVectors(self):
        "override base class"
        yawRadian = math.radians(self.yaw)
        yawCos = math.cos(yawRadian)
        pitchRadian = math.radians(self.pitch)
        pitchCos = math.cos(pitchRadian)
        frontX = yawCos * pitchCos
        frontY = math.sin(pitchRadian)
        frontZ = math.sin(yawRadian) * pitchCos
        self.front = QVector3D(frontX, frontY, frontZ)
        self.front.normalize()
        self.right = self.front.crossProduct(self.worldUp)
        self.right.normalize()
        self.up = self.right.crossProduct(self.front)
        self.up.normalize()

    def move2pos(self, direction: str, deltaTime: float):
        "compute new position in given direction"
        velocity = self.movementSpeed * deltaTime
        direction = direction.lower()
        pos = QVector3D()
        pos.setX(self.position.x())
        pos.setY(self.position.y())
        pos.setZ(self.position.z())
        if direction not in self.availableMoves:
            raise ValueError(
                "Unknown direction {0}, available moves are {1}".format(
                    direction, self.availableMoves
                )
            )
        if direction == "+z":
            pos += self.front * velocity
        elif direction == "-z":
            pos -= self.front * velocity
        elif direction == "+x":
            pos += self.right * velocity
        elif direction == "-x":
            pos -= self.right * velocity
        elif direction == "+y":
            pos += self.up * velocity
        elif direction == "-y":
            pos -= self.up * velocity
        return pos

    def setWorldUp(self, wup: QVector3D):
        "world up"
        self.worldUp = wup
        self.updateVectors()

    def setPosition(self, pos: QVector3D):
        "set position"
        self.position = pos
        self.updateVectors()
