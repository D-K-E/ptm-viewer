# author: Kaan Eraslan
# license: see, LICENSE
# No warranties, see LICENSE
import numpy as np


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
    assert arr1.shape == arr2.shape
    newarr = np.sum(arr1 * arr2, axis=1, dtype=np.float32)
    return newarr


def get_matrix_to_vector_dot(mat: np.ndarray,
                             vec: np.ndarray):
    "Get vector dot for each segment of matrix"
    mshape = mat[0, :].shape
    assert mshape == vec.shape
    d1 = get_vector_dot(mat[0, :], vec)
    d2 = get_vector_dot(mat[1, :], vec)
    d3 = get_vector_dot(mat[2, :], vec)
    newmat = np.empty(mshape)
    newmat[:, 0] = d1
    newmat[:, 1] = d2
    newmat[:, 2] = d3
    return newmat


def factor_3colmat_with_vec(mat: np.ndarray,
                            vec: np.ndarray):
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
    xdist = (apoint.x - xarr)**2
    ydist = (apoint.y - yarr)**2
    return np.sqrt(xdist + ydist)


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
    newimage = np.uint8(np.interp(newimage,
                                  (newimage.min(),
                                   newimage.max()),
                                  (0, 255))
                        )
    newimage = newimage.reshape(imshape)
    return ImageArray(newimage)
