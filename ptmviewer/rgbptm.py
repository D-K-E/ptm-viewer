# author: Kaan Eraslan
# license: see, LICENSE
# No warranties, see LICENSE

import pdb
import numpy as np

from ptmviewer.ptmparser import PTMFileParse
from ptmviewer.utils.utils import getInterpolationTable

from ctypes import c_float, c_uint
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
            [luvec ** 2, lvvec ** 2, lvvec * luvec, luvec, lvvec, np.ones_like(lvvec),],
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
        intensity = np.squeeze(intensity.reshape((self.imheight, self.imwidth, -1)))
        return intensity

    def form_surface_normal(self, luvec, lvvec):
        "Form surface normal matrice"
        normal = np.array(
            [luvec, lvvec, np.sqrt(1 - (luvec ** 2) - (lvvec ** 2))], dtype=np.float,
        )
        return normal.T

    def get_surface_normal(self, coeffarr):
        """
        Surface normal
        """
        luvec = self.get_light_dirU_vec(coeffarr)
        lvvec = self.get_light_dirV_vec(coeffarr)
        return self.form_surface_normal(luvec, lvvec)

    def setSurfaceNormal(self) -> None:
        "Set surface normal values to ptm"
        coeffs = self.imcoeffs.reshape((3, -1, 6))
        normals = np.empty((3, coeffs.shape[1], 3), dtype=np.float)
        normalR = self.get_surface_normal(coeffs[0, :, :])
        normalG = self.get_surface_normal(coeffs[1, :, :])
        normalB = self.get_surface_normal(coeffs[2, :, :])
        normals[0, :, :] = normalR
        normals[1, :, :] = normalG
        normals[2, :, :] = normalB
        self.normal = normals
        return

    def interpolate_normal(self, normal: np.ndarray):
        "interpolate normals"
        normal[:, 0] = np.interp(
            normal[:, 0], (normal[:, 0].min(), normal[:, 0].max()), (0, 255)
        )
        normal[:, 1] = np.interp(
            normal[:, 1], (normal[:, 1].min(), normal[:, 1].max()), (0, 255)
        )
        normal[:, 2] = np.interp(
            normal[:, 2], (normal[:, 2].min(), normal[:, 2].max()), (0, 255)
        )
        normalMap = normal.reshape((self.imheight, self.imwidth, 3))
        nmap = normalMap.astype("uint8", copy=False)
        return nmap

    def getChannelNormalMap(self, channel: str):
        "get normal map for surface normals per channel"
        channel = channel.lower()
        if channel == "r" or channel == "red":
            normal = self.normal[0, :, :]
        elif channel == "g" or channel == "green":
            normal = self.normal[1, :, :]
        elif channel == "b" or channel == "blue":
            normal = self.normal[2, :, :]
        nshape = normal.shape
        nmap = self.interpolate_normal(normal)
        return Image.fromarray(nmap)

    def getNormalMaps(self):
        "get normal map"
        if not self.normal:
            self.setSurfaceNormal()
        nmapR = self.getChannelNormalMap("red")
        nmapG = self.getChannelNormalMap("green")
        nmapB = self.getChannelNormalMap("blue")
        return nmapR, nmapG, nmapB

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
        if self.image:
            return Image.fromarray(self.image)
        self.setImage()
        return Image.fromarray(self.image)

    def getNbVertices(self):
        """
        obtain vertices from coefficients suitable for opengl drawing

        The idea is to create the content of the vbo here, than pass it 
        directly for rendering.
        The two for loops are heavy but necessary for stocking the
        coordinates, since they themselves play a role in rendering.

        We split the coefficients array into channels.
        Then reshape the resulting channel coefficients into the image shape
        The reason is that rgbptm shader takes in a position, and 18
        coefficients to come up with a fragment color.
        Due to our specification of the memory layout, the first three
        components of the VAO has to be position in 3d coordinates
        """
        rowsize = self.imwidth * self.imheight
        warr = np.array(range(self.imwidth), dtype=c_float)
        wnorm = np.interp(warr, (warr.min(), warr.max()), (-1, 1))
        harr = np.array(range(self.imheight), dtype=c_float)
        hnorm = np.interp(harr, (harr.min(), harr.max()), (-1, 1))
        #
        vertices = np.empty((self.imheight, self.imwidth, 21), dtype=c_float)
        # add width arr to vertices
        vertices = vertices.reshape(self.imwidth, 21 * self.imheight, 1)
        vertices[:, 0, 0] = wnorm
        vertices = vertices.reshape(self.imheight, 21 * self.imwidth, 1)
        vertices[:, 1, 0] = hnorm
        vertices = vertices.reshape(rowsize, 21, 1)
        vertices[:, 2, 0] = 1.0
        vertices = vertices.reshape(rowsize, 21)
        indices = np.array([i for i in range(rowsize)], dtype=c_uint)

        rcoeff = self.imcoeffs[0, :, :]
        gcoeff = self.imcoeffs[1, :, :]
        bcoeff = self.imcoeffs[2, :, :]
        vertices[:, 3:9] = rcoeff
        vertices[:, 9:15] = gcoeff
        vertices[:, 15:] = bcoeff
        vnb = vertices.shape[0]
        return vertices.reshape(-1), vnb
