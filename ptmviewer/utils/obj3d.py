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

from abc import ABC, abstractmethod


class AbstractRigid3dObject(ABC):
    "Abstract rigid 3d object"

    def __init__(self):
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.front = None
        self.worldUp = {"x": 0.0, "y": 1.0, "z": 0.0}
        # up with respect to center of the world
        self.up = None  # with respect to the objects current position
        self.right = None  # with respect to the objects current position

        # Euler angles for rotation
        self.yaw = -90.0  # rotation over z axis
        self.pitch = 0.0  # rotation over y axis
        self.roll = 0.0  # rotation over x axis

        # movement
        self.movementSensitivity = 0.001
        self.movementSpeed = 2.5
        self.availableMoves = ["+z", "-z", "+x", "-x", "+y", "-y"]

    @property
    @abstractmethod
    def z_axis_rotation_matrix(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def y_axis_rotation_matrix(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def x_axis_rotation_matrix(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def rotation_matrix(self):
        raise NotImplementedError

    @abstractmethod
    def update_vectors(self):
        raise NotImplementedError

    @abstractmethod
    def move(self, direction: str, deltaTime: float):
        raise NotImplementedError

    @abstractmethod
    def set_position(self, pos):
        raise NotImplementedError

    @abstractmethod
    def set_world_up(self, pos):
        raise NotImplementedError

    def check_angle(self, angle: float, angle_name: str):
        "check angle if it is in correct type and value"
        if isinstance(angle, float) is False:
            raise TypeError(angle_name + " value must be float: " + str(type(angle)))
        if angle < -360.0:
            raise ValueError(
                angle_name + " value can not be lower than -360: " + str(angle)
            )
        if angle > 360.0:
            raise ValueError(
                angle_name + " value can not be higher than 360: " + str(angle)
            )

    def set_yaw(self, val: float):
        "Set yaw value"
        self.check_angle(angle=val, angle_name="yaw")
        self.update_vectors()

    def set_pitch(self, val: float):
        "Set yaw value"
        self.check_angle(angle=val, angle_name="pitch")
        self.update_vectors()

    def set_roll(self, val: float):
        "Set yaw value"
        self.check_angle(angle=val, angle_name="roll")
        self.update_vectors()


class PureRigid3dObject(AbstractRigid3dObject):
    "Basic rigid body in 3d world"

    def __init__(self):
        self.position = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }  # object at the center of the world
        self.front = None
        self.worldUp = {"x": 0.0, "y": 1.0, "z": 0.0}
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

    def update_vectors(self):
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
        self.front = {"x": self.front[0], "y": self.front[1], "z": self.front[2]}
        self.right = {"x": self.right[0], "y": self.right[1], "z": self.right[2]}
        self.up = {"x": self.up[0], "y": self.up[1], "z": self.up[2]}

    @staticmethod
    def translateVec(
        axisVector: dict, positionVector: dict, velocity: float, isPlus=True
    ) -> dict:
        "Translate vector towards axis vector with velocity and position vector"
        atpl = tuple(axisVector.values())
        ptpl = tuple(positionVector.values())
        multip = scalar2vecMult(atpl, velocity)
        if isPlus:
            newPosTpl = vec2vecAdd(ptpl, multip)
        else:
            newPosTpl = vec2vecSubs(ptpl, multip)
        return {"x": newPosTpl[0], "y": newPosTpl[1], "z": newPosTpl[2]}

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
            position = self.translateVec(
                axisVector=self.front,
                positionVector=self.position,
                velocity=velocity,
                isPlus=True,
            )
        elif direction == "-z":
            position = self.translateVec(
                axisVector=self.front,
                positionVector=self.position,
                velocity=velocity,
                isPlus=False,
            )
        elif direction == "+x":
            position = self.translateVec(
                axisVector=self.right,
                positionVector=self.position,
                velocity=velocity,
                isPlus=True,
            )
        elif direction == "-x":
            position = self.translateVec(
                axisVector=self.right,
                positionVector=self.position,
                velocity=velocity,
                isPlus=False,
            )
        elif direction == "+y":
            position = self.translateVec(
                axisVector=self.up,
                positionVector=self.position,
                velocity=velocity,
                isPlus=True,
            )
        elif direction == "-y":
            position = self.translateVec(
                axisVector=self.up,
                positionVector=self.position,
                velocity=velocity,
                isPlus=False,
            )

        return position

    def move(self, direction: str, deltaTime: float):
        "move object to its new position"
        self.position = self.move2pos(direction, deltaTime)
        self.update_vectors()

    def check_coordinate_proc(self, pos: dict):
        "check coordinate type and value"
        if not isinstance(pos, dict):
            raise TypeError("Given coordinates are not in type dict: " + str(type(pos)))
        pkeys = list(pos.keys())
        if not all([True for k in pkeys if k in ["x", "y", "z"]]):
            mess = "Given coordinates do not have x, y, z."
            mess += " It has: " + str(pkeys)
            raise ValueError(mess)
        pvals = list(pos.values())
        if not all([isinstance(v, float) for v in pvals]):
            raise TypeError("Given coordinates do not have proper type float")

    def set_world_up(self, wup: dict):
        "world up"
        self.check_coordinate_proc(wup)
        self.worldUp = wup
        self.update_vectors()

    def set_position(self, pos: dict):
        "set position"
        self.check_coordinate_proc(pos)
        self.position = pos
        self.update_vectors()


class QtRigid3dObject(AbstractRigid3dObject):
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

    def update_vectors(self):
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

    def move(self, direction: str, deltaTime: float):
        "move object to its new position"
        self.position = self.move2pos(direction, deltaTime)
        self.update_vectors()

    def check_coordinate_proc(self, pos: QVector3D):
        "check coordinate type and value"
        if not isinstance(pos, QVector3D):
            raise TypeError(
                "Given coordinates are not in type QVector3D: " + str(type(pos))
            )
        pvals = [pos.x(), pos.y(), pos.z()]
        if not all([isinstance(v, float) for v in pvals]):
            raise TypeError("Given coordinates do not have proper type float")

    def set_world_up(self, wup: QVector3D):
        "world up"
        self.check_coordinate_proc(wup)
        self.worldUp = wup
        self.update_vectors()

    def set_position(self, pos: QVector3D):
        "set position"
        self.check_coordinate_proc(pos)
        self.position = pos
        self.update_vectors()
