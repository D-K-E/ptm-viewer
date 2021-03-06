# author: Kaan Eraslan
# license: see, LICENSE
# No warranties, see LICENSE
import numpy as np
from PySide2.QtGui import QVector3D
from PySide2.QtGui import QVector4D
from PySide2.QtGui import QMatrix4x4

from typing import Tuple, List, Dict
import math


def normalize_1d_array(arr):
    "Normalize 1d array"
    assert arr.ndim == 1
    result = None
    if np.linalg.norm(arr) == 0:
        result = arr
    else:
        result = arr / np.linalg.norm(arr)
    return result


def normalize_3col_array(arr):
    "Normalize 3 column array"
    assert arr.shape[1] == 3
    assert arr.ndim == 2
    normal = np.copy(arr)
    normal[:, 0] = normalize_1d_array(normal[:, 0])
    normal[:, 1] = normalize_1d_array(normal[:, 1])
    normal[:, 2] = normalize_1d_array(normal[:, 2])
    return normal


def get_vector_dot(arr1, arr2):
    "Get vector dot product for 2 matrices"
    if arr1.shape != arr2.shape:
        raise ValueError("arr1 and arr2 shape should be same")
    newarr = np.sum(arr1 * arr2, axis=1, dtype=np.float32)
    return newarr


def get_matrix_to_vector_dot(mat: np.ndarray, vec: np.ndarray):
    "Get vector dot for each segment of matrix"
    mshape = mat[0, :].shape
    if mshape != vec.shape:
        raise ValueError("Matrix vector shape should be same with vector shape")
    d1 = get_vector_dot(mat[0, :], vec)
    d2 = get_vector_dot(mat[1, :], vec)
    d3 = get_vector_dot(mat[2, :], vec)
    newmat = np.empty(mshape)
    newmat[:, 0] = d1
    newmat[:, 1] = d2
    newmat[:, 2] = d3
    return newmat


def factor_3colmat_with_vec(mat: np.ndarray, vec: np.ndarray):
    "Factor matrix columns with vector"
    assert vec.ndim == 1
    assert mat.shape[1] == 3
    assert vec.shape[0] == mat.shape[0]
    vec = np.where(vec != 0, vec, 0.00001)  # against zero divisions
    newmat = np.empty_like(mat)
    newmat[:, 0] = mat[:, 0] / vec
    newmat[:, 1] = mat[:, 1] / vec
    newmat[:, 2] = mat[:, 2] / vec
    return newmat


def getDistancePoint2Array(apoint, coordarr):
    "Get distance between point1 and point2"
    yarr = coordarr[:, 0]  # row nb
    xarr = coordarr[:, 1]  # col nb
    xdist = (apoint.x - xarr) ** 2
    ydist = (apoint.y - yarr) ** 2
    return np.sqrt(xdist + ydist)


def getInterpolationTable(arr: np.ndarray, mapRange: Tuple[float, float]) -> dict:
    "Interpolate given one dimensional array into given range output as a table"
    assert arr.ndim == 1
    newarr = np.interp(arr, (arr.min(), arr.max()), mapRange)
    return {arr[i]: newarr[i] for i in range(arr.size)}


class ImageArray:
    "Image array have some additional properties besides np.ndarray"

    def __init__(self, image: np.ndarray):
        assert isinstance(image, np.ndarray)
        self.image = image

    @property
    def norm_coordinates(self):
        "Get normalized coordinates of the image pixels"
        # pdb.set_trace()
        rownb, colnb = self.image.shape[0], self.image.shape[1]
        norm = np.empty_like(self.coordinates, dtype=np.float32)
        norm[:, 0] = self.coordinates[:, 0] / rownb
        norm[:, 1] = self.coordinates[:, 1] / colnb
        return norm

    @property
    def norm_image(self):
        "Get normalized image with pixel values divided by 255"
        return self.image / 255.0

    @property
    def coordinates(self):
        "Coordinates of the image pixels"
        rownb, colnb = self.image.shape[:2]
        coords = [[(row, col) for col in range(colnb)] for row in range(rownb)]
        coordarray = np.array(coords)
        return coordarray.reshape((-1, 2))

    @property
    def arrshape(self):
        "get array shape"
        return self.image.shape

    @property
    def flatarr(self):
        "get flattened array"
        return self.image.flatten()


def interpolateImage(imarr: ImageArray):
    "Interpolate image array"
    imshape = imarr.image.shape
    newimage = imarr.image.flatten()
    newimage = np.uint8(np.interp(newimage, (newimage.min(), newimage.max()), (0, 255)))
    newimage = newimage.reshape(imshape)
    return ImageArray(newimage)


def normalize_tuple(vec: tuple):
    "Normalize 1 d tuple"
    vecSum = sum([v ** 2 for v in vec])
    if vecSum == 0:
        return vec
    else:
        return tuple([v / vecSum for v in vec])


def crossProduct(vec1, vec2):
    "take cross products of two vectors"
    assert len(vec1) == 3 and len(vec2) == 3
    vec3x = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    vec3y = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    vec3z = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return (vec3x, vec3y, vec3z)


def vec2vecDot(vec1, vec2):
    "vector to vector dot product"
    assert len(vec1) == len(vec2)
    return tuple(sum(v1 * v2 for v1, v2 in zip(vec1, vec2)))


def sliceCol(colInd: int, matrix: list):
    "slice column values from matrix"
    rownb = len(matrix)
    return [matrix[i, colInd] for i in range(rownb)]


def set_column_mat(pos: int, arr: list, mat: list) -> list:
    "set column to matrix at given position"
    if len(mat) != len(arr):
        raise ValueError("Col height not equal to size of array to insert")
    for index, row in enumerate(mat):
        row[pos] = arr[index]
    return mat


def mat2matDot(mat1: list, mat2: list) -> list:
    "Dot product in pure python"
    if len(mat1[0]) != len(mat2):
        raise ValueError("mat1 row size is not equal mat2 column size")
    colnb = len(mat1[0])
    mat = []
    for rown in range(len(mat1)):
        newmatRow = []
        mat1Row = mat1[rown]
        for coln in range(colnb):
            mat2col = sliceCol(coln, mat2)
            newmatRow.append(vec2vecDot(mat1Row, mat2col))
        mat.append(newmatRow)
    return mat


def mat2vecDot(mat: list, vec: list) -> list:
    "dot product in pure python matrix to vector"
    if len(mat[0]) != len(vec):
        raise ValueError("Matrix vector shape should be same with vector shape")
    newmat = [[0 for i in range(len(mat[0]))] for k in range(len(mat))]
    for i, row in enumerate(mat):
        newrow = vec2vecDot(row, vec)
        newmat = set_column_mat(pos=i, arr=newrow, mat=newmat)
    return newmat


def scalar2vecMult(vec, scalar):
    "scalar multiplication of a vector"
    return tuple([v * scalar for v in vec])


def vec2vecAdd(vec1, vec2):
    "vector to vector addition"
    assert len(vec1) == len(vec2)
    return tuple([vec1[i] + vec2[i] for i in range(len(vec1))])


def vec2vecSubs(vec1, vec2):
    "vector to vector subtraction"
    assert len(vec1) == len(vec2)
    return tuple([vec1[i] - vec2[i] for i in range(len(vec1))])


def computePerspectiveNp(fieldOfView: float, aspect: float, zNear: float, zFar: float):
    "Reproduces glm perspective function"
    assert aspect != 0
    assert zNear != zFar
    fieldOfViewRad = np.radians(fieldOfView)
    fieldHalfTan = np.tan(fieldOfViewRad / 2)
    # mat4
    result = np.zeros((4, 4), dtype=float)
    result[0, 0] = 1 / (aspect * fieldHalfTan)
    result[1, 1] = 1 / fieldHalfTan
    result[2, 2] = -(zFar + zNear) / (zFar - zNear)
    result[3, 2] = -1
    result[2, 3] = -(2 * zFar * zNear) / (zFar - zNear)
    return result


def computePerspectiveQt(fieldOfView: float, aspect: float, zNear: float, zFar: float):
    "matrice"
    mat = QMatrix4x4(*[0.0 for i in range(16)])
    return mat.perspective(fieldOfView, aspect, zNear, zFar)


def computeLookAtPure(pos: tuple, target: tuple, worldUp: tuple) -> list:
    ""
    assert len(pos) == 3 and len(target) == 3
    assert len(worldUp) == 3
    zaxis = normalize_tuple(vec2vecSubs(pos, target))

    # x axis
    normWorld = normalize_tuple(worldUp)
    xaxis = normalize_tuple(crossProduct(normWorld, zaxis))
    yaxis = crossProduct(zaxis, xaxis)
    translation = [[1 for i in range(4)] for k in range(4)]
    translation[0][3] = -pos[0]
    translation[1][3] = -pos[1]  # third col, second row
    translation[2][3] = -pos[2]

    rotation = [[1 for i in range(4)] for k in range(4)]
    rotation[0][0] = xaxis[0]
    rotation[0][1] = xaxis[1]
    rotation[0][2] = xaxis[2]
    rotation[1][0] = yaxis[0]
    rotation[1][1] = yaxis[1]
    rotation[1][2] = yaxis[2]
    rotation[2][0] = zaxis[0]
    rotation[2][1] = zaxis[1]
    rotation[2][2] = zaxis[2]
    return mat2matDot(translation, rotation)


def computeLookAtMatrixNp(
    position: np.ndarray, target: np.ndarray, worldUp: np.ndarray
):
    "Compute a look at matrix for given position and target"
    assert position.ndim == 1 and target.ndim == 1 and worldUp.ndim == 1
    zaxis = normalize_1d_array(position - target)

    # positive xaxis at right
    xaxis = normalize_1d_array(np.cross(normalize_1d_array(worldUp), zaxis))
    # camera up
    yaxis = np.cross(zaxis, xaxis)

    # compute translation matrix
    translation = np.ones((4, 4), dtype=np.float)
    translation[0, 3] = -position[0]  # third col, first row
    translation[1, 3] = -position[1]  # third col, second row
    translation[2, 3] = -position[2]

    # compute rotation matrix
    rotation = np.ones((4, 4), dtype=np.float)
    rotation[0, 0] = xaxis[0]
    rotation[0, 1] = xaxis[1]
    rotation[0, 2] = xaxis[2]
    rotation[1, 0] = yaxis[0]
    rotation[1, 1] = yaxis[1]
    rotation[1, 2] = yaxis[2]
    rotation[2, 0] = zaxis[0]
    rotation[2, 1] = zaxis[1]
    rotation[2, 2] = zaxis[2]

    return np.dot(translation, rotation)


def computeLookAtMatrixQt(position: np.ndarray, target: np.ndarray, up: np.ndarray):
    "look at matrice"
    eye = QVector3D(position[0], position[1], position[2])
    target = QVector3D(target[0], target[1], target[2])
    upvec = QVector3D(up[0], up[1], up[2])
    mat4 = QMatrix4x4()
    return mat4.lookAt(eye, target, upvec)


def arr2vec(arr: np.ndarray):
    "convert array 2 vector"
    sqarr = np.squeeze(arr)
    assert sqarr.size == 4
    return QVector4D(sqarr[0], sqarr[1], sqarr[2], sqarr[3])


def arr2qmat(arr: np.ndarray):
    "array to matrix 4x4"
    assert arr.shape == (4, 4)
    mat4 = QMatrix4x4()
    for rowNb in range(arr.shape[0]):
        rowarr = arr[rowNb, :]
        rowvec = arr2vec(rowarr)
        mat4.setRow(rowNb, rowvec)
    #
    return mat4


def move3dObjPure(
    direction: str,
    positionVector: Tuple[float, float, float],
    xaxis: Tuple[float, float, float],
    yaxis: Tuple[float, float, float],
    zaxis: Tuple[float, float, float],
    deltaTime: float,
    speed: float,
    availableMoves=["+z", "-z", "-x", "+x", "+y", "-y"],
):
    ""
    velocity = speed * deltaTime
    direction = direction.lower()
    if direction not in availableMoves:
        raise ValueError(
            "Unknown direction {0}, available moves are {1}".format(
                direction, availableMoves
            )
        )
    if direction == "+z":
        multip = scalar2vecMult(zaxis, velocity)
        positionVector = vec2vecAdd(positionVector, multip)
    elif direction == "-z":
        multip = scalar2vecMult(zaxis, velocity)
        positionVector = vec2vecSubs(positionVector, multip)
    elif direction == "+x":
        multip = scalar2vecMult(xaxis, velocity)
        positionVector = vec2vecAdd(positionVector, multip)
    elif direction == "-x":
        multip = scalar2vecMult(xaxis, velocity)
        positionVector = vec2vecSubs(positionVector, multip)
    elif direction == "+y":
        multip = scalar2vecMult(yaxis, velocity)
        positionVector = vec2vecAdd(positionVector, multip)
    elif direction == "-y":
        multip = scalar2vecMult(yaxis, velocity)
        positionVector = vec2vecSubs(positionVector, multip)

    return positionVector


def move3dObjQt(
    direction: str,
    positionVector: QVector3D,
    xaxis: QVector3D,
    yaxis: QVector3D,
    zaxis: QVector3D,
    deltaTime: float,
    speed: float,
    availableMoves=["+x", "-x", "+y", "-y", "+z", "-z"],
):
    ""
    velocity = speed * deltaTime
    direction = direction.lower()
    if direction not in availableMoves:
        raise ValueError(
            "Unknown direction {0}, available moves are {1}".format(
                direction, availableMoves
            )
        )
    if direction == "+x":
        positionVector += xaxis * velocity
    elif direction == "-x":
        positionVector -= xaxis * velocity
    elif direction == "+y":
        positionVector += yaxis * velocity
    elif direction == "-y":
        positionVector -= yaxis * velocity
    elif direction == "+z":
        positionVector += zaxis * velocity
    elif direction == "-z":
        positionVector -= zaxis * velocity
    return positionVector


def computeFrontRightPure(yaw: float, pitch: float, worldUp=(0.0, 1.0, 0.0)):
    "Compute front vector"
    yawRadian = math.radians(yaw)
    yawCos = math.cos(yawRadian)
    pitchRadian = math.radians(pitch)
    pitchCos = math.cos(pitchRadian)
    frontX = yawCos * pitchCos
    frontY = math.sin(pitchRadian)
    frontZ = math.sin(yawRadian) * pitchCos
    front = (frontX, frontY, frontZ)
    front = normalize_tuple(front)
    right = crossProduct(front, worldUp)
    right = normalize_tuple(right)
    up = crossProduct(right, front)
    up = normalize_tuple(up)
    return (front, right, up)


def computeFrontRightQt(yaw: float, pitch: float, worldUp=QVector3D(0.0, 1.0, 0.0)):
    ""
    yawRadian = math.radians(yaw)
    yawCos = math.cos(yawRadian)
    pitchRadian = math.radians(pitch)
    pitchCos = math.cos(pitchRadian)
    frontX = yawCos * pitchCos
    frontY = math.sin(pitchRadian)
    frontZ = math.sin(yawRadian) * pitchCos
    front = QVector3D(frontX, frontY, frontZ)
    front.normalize()
    right = QVector3D.crossProduct(front, worldUp)
    right.normalize()
    up = QVector3D.crossProduct(right, front)
    up.normalize()
    return (front, right, up)
