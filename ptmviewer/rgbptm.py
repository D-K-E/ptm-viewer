# author: Kaan Eraslan
# license: see, LICENSE
# No warranties, see LICENSE

from ptmviewer.ptmparser import PTMFileParse

import numpy as np
from PIL import Image


class RGBPTM(PTMFileParse):
    "RGB ptm file handler"

    def __init__(self, ptmpath: str):
        super().__init__(ptmpath)
        self.ptmfile = self.parse()
        self.imheight = self.ptmfile["image_height"]
        self.imwidth = self.ptmfile["image_width"]
        self.coefficients = self.ptmfile["coeffarr"]
        self.imcoeffs = self.coefficients.reshape((3, self.imheight * self.imwidth, 6))
        self.normal = None
        self.image = None
        self.setSurfaceNormal()
        self.setImage()

    def get_light_dirU_vec(self, coeffarr: np.ndarray):
        """
        Get light direction U vector using formula:
        l_u0 = (a_2 * a_4 - 2 * a_1 * a_3) / (4 * a_0 * a_1 - a_2**2)
        """
        nomin = (coeffarr[:, 2] * coeffarr[:, 4]) - (
            2 * coeffarr[:, 1] * coeffarr[:, 3]
        )
        denomin = (4 * coeffarr[:, 0] * coeffarr[:, 1]) - (coeffarr[:, 2] ** 2)

        newcoeffarr = nomin / denomin
        return newcoeffarr

    def get_light_dirV_vec(self, coeffarr: np.ndarray):
        """
        Get light direction U vector using formula:
        l_v0 = (a_2 * a_3 - 2 * a_0 * a_4) / (4 * a_0 * a_1 - a_2**2)
        """
        nomin = (coeffarr[:, 2] * coeffarr[:, 3]) - (
            2 * coeffarr[:, 0] * coeffarr[:, 4]
        )
        denomin = (4 * coeffarr[:, 0] * coeffarr[:, 1]) - (coeffarr[:, 2] ** 2)
        newcoeffarr = nomin / denomin
        return newcoeffarr

    def form_light_direction_mat(self, luvec, lvvec):
        "Form the light direction matrice"
        l_mat = np.array(
            [luvec ** 2, lvvec ** 2, lvvec * luvec, luvec, lvvec, np.ones_like(lvvec)],
            dtype=np.float,
        )
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

    def setSurfaceNormal(self):
        "Set surface normal values to ptm"
        rnorm = self.get_surface_normal(self.imcoeffs[0, :, :])
        gnorm = self.get_surface_normal(self.imcoeffs[1, :, :])
        bnorm = self.get_surface_normal(self.imcoeffs[2, :, :])
        nshape = rnorm.shape
        normals = np.empty((3, *nshape), dtype=np.float64)
        normals[0, :] = rnorm
        normals[1, :] = gnorm
        normals[2, :] = bnorm
        self.normal = normals
        return normals

    def getNormalMap(self):
        "get normal map for surface normals"
        nr = self.normal[0, :]
        ng = self.normal[1, :]
        nb = self.normal[2, :]
        nshape = nr.shape
        normals = np.empty((3, *nshape), dtype=np.uint8)
        norm1 = np.interp(nr, (nr.min(), nr.max()), (0, 255))
        norm2 = np.interp(ng, (ng.min(), ng.max()), (0, 255))
        norm3 = np.interp(nb, (nb.min(), nb.max()), (0, 255))
        normals[0, :] = norm1
        normals[1, :] = norm2
        normals[2, :] = norm3
        return Image.fromarray(normals)

    def setImage(self):
        "set image rgb values"
        image = np.empty((self.imheight, self.imwidth, 3), dtype=np.uint8)
        blue = self.get_channel_intensity(self.imcoeffs[2, :])
        red = self.get_channel_intensity(self.imcoeffs[0, :])
        green = self.get_channel_intensity(self.imcoeffs[1, :])
        image[:, :, 0] = np.interp(red, (red.min(), red.max()), (0, 255))
        image[:, :, 1] = np.interp(green, (green.min(), green.max()), (0, 255))
        image[:, :, 2] = np.interp(blue, (blue.min(), blue.max()), (0, 255))
        self.image = image
        return

    def getImage(self):
        return Image.fromarray(self.image)

