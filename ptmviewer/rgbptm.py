# author: Kaan Eraslan
# license: see, LICENSE
# No warranties, see LICENSE
import numpy as np

from utils import normalize_1d_array
from utils import ImageArray
from utils import interpolateImage


class RGBPTM:
    "Regroups methods on rgb ptm"

    def __init__(self,
                 coeffarr: np.ndarray,
                 image_height: int,
                 image_width: int,
                 scales: [float],
                 biases: [int]):
        self.arr = np.copy(coeffarr)
        self.scales = scales
        self.biases = biases
        self.imheight = image_height
        self.imwidth = image_width

    def copy(self):
        "Copy self"
        return RGBPTM(coeffarr=np.copy(self.arr),
                      image_height=self.imheight,
                      image_width=self.imwidth,
                      scales=self.scales,
                      biases=self.biases)

    @property
    def imcoeffarr(self):
        "Image coefficients array"
        return self.arr.reshape((3,
                                 self.imheight * self.imwidth,
                                 6))

    @property
    def red_channel_coefficients(self):
        "Red channel coefficients array"
        return self.imcoeffarr[0, :, :]

    @property
    def green_channel_coefficients(self):
        "Green channel coefficients array"
        return self.imcoeffarr[1, :, :]

    @property
    def blue_channel_coefficients(self):
        "Blue channel coefficients array"
        return self.imcoeffarr[2, :, :]

    @property
    def red_light_dirU_vec(self):
        "Red light direction u vector"
        return self.get_light_dirU_vec(self.red_channel_coefficients)

    @property
    def red_light_dirV_vec(self):
        "Red light direction v vector"
        return self.get_light_dirV_vec(self.red_channel_coefficients)

    @property
    def red_channel_surface_normal(self):
        "Red channel surface normal"
        return self.form_surface_normal(self.red_light_dirU_vec,
                                        self.red_light_dirV_vec)

    @property
    def red_channel_pixel_values(self):
        "Get red channel pixel values"
        return self.get_channel_intensity(self.red_channel_coefficients)

    @property
    def red_channel_normalized_pixel_values(self):
        flat = self.red_channel_pixel_values.flatten()
        return normalize_1d_array(flat)

    @property
    def green_light_dirU_vec(self):
        "Green light direction u vector"
        return self.get_light_dirU_vec(self.green_channel_coefficients)

    @property
    def green_light_dirV_vec(self):
        "Green light direction v vector"
        return self.get_light_dirV_vec(self.green_channel_coefficients)

    @property
    def green_channel_surface_normal(self):
        "Green channel surface normal"
        return self.form_surface_normal(self.green_light_dirU_vec,
                                        self.green_light_dirV_vec)

    @property
    def green_channel_pixel_values(self):
        "Get green channel pixel values"
        return self.get_channel_intensity(self.green_channel_coefficients)

    @property
    def green_channel_normalized_pixel_values(self):
        flat = self.green_channel_pixel_values.flatten()
        return normalize_1d_array(flat)

    @property
    def blue_light_dirU_vec(self):
        "Blue light direction u vector"
        return self.get_light_dirU_vec(self.blue_channel_coefficients)

    @property
    def blue_light_dirV_vec(self):
        "Blue light direction v vector"
        return self.get_light_dirV_vec(self.blue_channel_coefficients)

    @property
    def blue_channel_surface_normal(self):
        "Green channel surface normal"
        return self.form_surface_normal(self.blue_light_dirU_vec,
                                        self.blue_light_dirV_vec)

    @property
    def blue_channel_pixel_values(self):
        "Get blue channel pixel values"
        return self.get_channel_intensity(self.blue_channel_coefficients)

    @property
    def blue_channel_normalized_pixel_values(self):
        flat = self.blue_channel_pixel_values.flatten()
        return normalize_1d_array(flat)

    @property
    def surface_normals(self):
        "Obtain surface normals"
        nshape = self.red_channel_surface_normal.shape
        normals = np.empty((3, *nshape), dtype=np.float32)
        normals[0, :] = self.red_channel_surface_normal
        normals[1, :] = self.green_channel_surface_normal
        normals[2, :] = self.blue_channel_surface_normal
        return normals

    @property
    def imarr(self):
        "Get image array"
        image = np.empty((self.imheight, self.imwidth, 3), dtype=np.float32)
        image[:, :, 0] = self.red_channel_pixel_values
        image[:, :, 1] = self.green_channel_pixel_values
        image[:, :, 2] = self.blue_channel_pixel_values
        # image = np.fliplr(image)  # flip image in column wise
        imarr = ImageArray(image)
        imarr = interpolateImage(imarr)
        return imarr

    @property
    def image(self):
        "Get image"
        return np.fliplr(self.imarr.image)

    def get_light_dirU_vec(self, coeffarr: np.ndarray):
        """
        Get light direction U vector using formula:
        l_u0 = (a_2 * a_4 - 2 * a_1 * a_3) / (4 * a_0 * a_1 - a_2**2)
        """
        nomin = (
            coeffarr[:, 2] * coeffarr[:, 4]
        ) - (
            2 * coeffarr[:, 1] * coeffarr[:, 3]
        )
        denomin = (
            4 * coeffarr[:, 0] * coeffarr[:, 1]
        ) - (coeffarr[:, 2]**2)

        newcoeffarr = nomin / denomin
        return newcoeffarr

    def get_light_dirV_vec(self, coeffarr: np.ndarray):
        """
        Get light direction U vector using formula:
        l_v0 = (a_2 * a_3 - 2 * a_0 * a_4) / (4 * a_0 * a_1 - a_2**2)
        """
        nomin = (
            coeffarr[:, 2] * coeffarr[:, 3]
        ) - (
            2 * coeffarr[:, 0] * coeffarr[:, 4]
        )
        denomin = (
            4 * coeffarr[:, 0] * coeffarr[:, 1]
        ) - (coeffarr[:, 2]**2)
        newcoeffarr = nomin / denomin
        return newcoeffarr

    def form_light_direction_mat(self, luvec, lvvec):
        "Form the light direction matrice"
        l_mat = np.array([luvec**2,
                          lvvec**2,
                          lvvec * luvec,
                          luvec,
                          lvvec,
                          np.ones_like(lvvec)], dtype=np.float)
        return l_mat.T

    def get_light_direction_matrix(self, coeffarr: np.ndarray):
        "Set light direction vector from coeffarr"
        luvec = self.get_light_dirU_vec(coeffarr)
        lvvec = self.get_light_dirV_vec(coeffarr)
        return self.form_light_direction_mat(luvec, lvvec)

    def get_channel_intensity(self, channel_coeffarr: np.ndarray):
        "Get channel intensity given channel coefficient array"
        arr = channel_coeffarr.reshape((-1, 6))
        light_dir_mat = self.get_light_direction_matrix(arr)
        intensity = np.sum(arr * light_dir_mat, axis=1, dtype=np.float32)
        intensity = np.squeeze(intensity.reshape((self.imheight, self.imwidth,
                                                  -1))
                               )
        return intensity

    def form_surface_normal(self, luvec,
                            lvvec):
        "Form surface normal matrice"
        normal = np.array(
            [luvec,
             lvvec,
             np.sqrt(1 - luvec**2 - lvvec**2)
             ],
            dtype=np.float)
        return np.transpose(normal, (1, 0))

    def get_surface_normal(self, coeffarr):
        """
        Surface normal
        """
        luvec = self.get_light_dirU_vec(coeffarr)
        lvvec = self.get_light_dirV_vec(coeffarr)
        return self.form_surface_normal(luvec, lvvec)
