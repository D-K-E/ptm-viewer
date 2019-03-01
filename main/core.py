# author: Kaan Eraslan
# license: see, LICENSE
# No Warranties, use at your own risk, see LICENSE

import numpy as np
import pdb
from matplotlib.colors import LightSource as Lsource
import matplotlib.pyplot as plt
import cv2


class PTMFileParse:
    """
    Parse .ptm files according to
    http://www.hpl.hp.com/research/ptm/downloads/PtmFormat12.pdf
    """

    def __init__(self, ptmpath: str):
        self.path = ptmpath
        with open(self.path, 'rb') as f:
            self.raw = f.readlines()
            self.raw = [raw_line for raw_line in self.raw if raw_line]
        self.header = self.raw[0].decode('utf-8').strip()
        self.format = self.raw[1].decode('utf-8').strip()
        if self.format != "PTM_FORMAT_RGB":
            raise ValueError('ptm format {0} not supported'.format(self.format)
                             )

    @property
    def image_width(self):
        return int(self.raw[2].decode('utf-8').strip())

    @property
    def image_height(self):
        return int(self.raw[3].decode('utf-8').strip())

    @property
    def scales(self):
        scales = self.raw[4].decode('utf-8').strip().split()
        return np.array([float(s) for s in scales], dtype=np.float32)

    @property
    def biases(self):
        biases = self.raw[5].decode('utf-8').strip().split()
        return np.array([int(b) for b in biases], dtype=np.int32)
        
    def get_coeffarr(self):
        "Get coefficients array from bytelist"
        if self.format == "PTM_FORMAT_RGB":
            bytelist = self.raw[6:]
            # bytelist = reversed(bytelist)
            bstr = b''.join(bytelist)
            bstr = bstr[::-1]  # reverses the byte string due to format
            flatarr = np.frombuffer(bstr, dtype=np.uint8)
            flatarr = flatarr.reshape((-1, 6))
            flatarr = self.get_final_coefficient(flatarr)
        else:
            raise ValueError(
                """
                Working with an unsupported format {0}.
                Only uncompressed PTM_FORMAT_RGB is supported
                """.format(self.format)
            )
        return flatarr

    def get_final_coefficient(self, coeffarr: np.ndarray):
        """
        Get final coefficient using:

        Coeff_final = (Coeff_raw - bias) * scale
        """
        newcoeffarr = (coeffarr - self.biases) * self.scales
        return newcoeffarr

    def parse(self) -> dict:
        "Parse document and give output as dict"
        flatarr = self.get_coeffarr()
        coeffarr = self.get_final_coefficient(flatarr)
        out = {}
        out['coeffarr'] = coeffarr
        out['scales'] = self.scales
        out['biases'] = self.biases
        out['image_width'] = self.image_width
        out['image_height'] = self.image_height
        out['format'] = self.format
        return out


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
        rownb, colnb = self.image.shape[:2]
        coords = [[(row, col) for col in range(colnb)] for row in range(rownb)]
        coordarray = np.array(coords)
        coordarray = coordarray.reshape((-1, 2))
        coordarray[:, 0] = coordarray[:, 0] / rownb
        coordarray[:, 1] = coordarray[:, 1] / colnb
        return coordarray

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


class LightSource:
    "Simple implementation of a light source"

    def __init__(self, x, y, z=None):
        "light source"
        self.x = x
        self.y = y
        if z is not None:
            assert isinstance(z, float)
            assert z > 0.0 and z < 1.0
        self.radius = z
        self.coords = [x, y, z]

    def getDistance2Arr(self, arr):
        "Get range of the light source"
        arrshape = arr.shape
        assert arrshape[1] == 2  # should be x,y array
        return getDistancePoint2Array(apoint=self,
                                      coordarr=arr)

    def getNormDistance2Arr(self, image_row_nb: int,
                            image_col_nb: int,
                            norm_coordinates: np.ndarray):
        normpoint = {'x': self.x / image_col_nb,
                     'y': self.y / image_row_nb}
        assert norm_coordinates.shape[1] == 2
        return getDistancePoint2Array(apoint=normpoint,
                                      coordarr=norm_coordinates)


class Shader:
    "Shader regroups shading methods operating on pixels"

    def __init__(self, image: np.ndarray,
                 light_source: LightSource):
        assert isinstance(image, np.ndarray)
        self.image = ImageArray(image)
        self.light_source = light_source

    def assignColor2Image(self,
                          factor: float,
                          coordarr):
        """
        Assign new colors to image using coordinate array

        The main idea is to assign to the pixels specified in the
        coordinate array a color that is indicated by the factor
        """
        image = self.image.image
        image_factored = np.copy(image).astype(np.float32) * factor
        newimage = np.zeros_like(image, dtype=np.float32)
        newimage[coordarr[:, 0],
                 coordarr[:, 1], :] = image_factored[coordarr[:, 0],
                                                     coordarr[:, 1], :]
        newimage = interpolateImage(newimage)
        return newimage

    def filterWithLightRange(self):
        "Filter from image coordinates that are not in light range"
        image = self.image.image
        coordarr = self.image.coordinates
        image_colnb = image.shape[1]
        distance = self.getDistance2Arr(coordarr)
        inRangeCondition = self.light_source.z * image_colnb > distance
        distanceMasked = np.where(inRangeCondition,
                                  distance,  # can't be negative
                                  -1)  # -1 to mark areas outside range
        inrangeCoords = np.argwhere(distanceMasked != -1)
        rowCoordinates = np.squeeze(inrangeCoords)
        newcoordarr = coordarr[rowCoordinates, :]
        return newcoordarr

    def filterImageSharp(self):
        "Filter coords outside of light range sharply"
        image = self.image.image
        coordarr = self.image.coordinates
        image_colnb = image.shape[1]
        newcoordarr = self.filterWithLightRange(image_colnb,
                                                image,
                                                coordarr)
        newimage = self.assignColor2Image(coordarray=newcoordarr,
                                          factor=1.0)
        return newimage

    def filterImageDistanceFactor(self):
        "Filter coords outside of light range with a factor of distance"
        coordarr = self.image.coordinates
        image_colnb = self.image.image.shape[1]
        distance = self.light_source.getDistance2Arr(arr=coordarr)
        factor = self.light_source.z * image_colnb
        factor = distance / factor
        factor = 1 - factor
        newcoord = self.filterWithLightRange()
        newimage = self.assignColor2Image(factor=factor,
                                          coordarr=newcoord)
        return newimage

    def shade(self,
              fn: lambda x: x,
              pixel) -> [int, int, int, int]:
        "shade pixel using function"
        newpixel = fn(pixel)
        return newpixel


class PTMHandler:
    """
    Implements methods for manipulating Polynomial Texture Mappings (PTMs)
    """

    def __init__(self,
                 coeffarr: np.ndarray,
                 image_height: int,
                 image_width: int,
                 scales: [float],
                 biases: [int],
                 azdeg=315,
                 altdeg=45
                 ):
        self.arr = np.copy(coeffarr)
        self.scales = scales
        self.biases = biases
        self.imheight = image_height
        self.imwidth = image_width
        # light source angle and direction
        self.light_source = Lsource(azdeg=azdeg, altdeg=altdeg)
        # see compute shading for variable names
        self.Ia = 1.0
        self.ka = 0.2
        self.c1 = 2
        self.c2 = 2
        self.c3 = 2
        self.dL = 4
        self.Ip = 1.0
        self.kd = 0.45
        self.od = 1
        self.ks = 0.40
        self.Os = 1
        self.n = 3.0

        # see change diffuse gain for variable name
        self.g = 0.4

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
    def red_channel_normalized_surface_normal(self):
        "Red channel normalized surface normal using vector norm of axes"
        return self.normalize_surface_normal(self.red_channel_surface_normal)

    @property
    def red_channel_pixel_values(self):
        "Get red channel pixel values"
        return self.get_channel_intensity(self.red_channel_coefficients)

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
    def green_channel_normalized_surface_normal(self):
        "Green channel normalized surface normal using vector norm of axes"
        return self.normalize_surface_normal(self.green_channel_surface_normal)
        
    @property
    def green_channel_pixel_values(self):
        "Get green channel pixel values"
        return self.get_channel_intensity(self.green_channel_coefficients)

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
    def blue_channel_normalized_surface_normal(self):
        "Red channel normalized surface normal using vector norm of axes"
        return self.normalize_surface_normal(self.blue_channel_surface_normal)

    @property
    def blue_channel_pixel_values(self):
        "Get blue channel pixel values"
        return self.get_channel_intensity(self.blue_channel_coefficients)

    @property
    def imarr(self):
        "Get image array"
        image = np.zeros((self.imheight, self.imwidth, 3), dtype=np.float32)
        image[:, :, 0] = self.red_channel_pixel_values
        image[:, :, 1] = self.green_channel_pixel_values
        image[:, :, 2] = self.blue_channel_pixel_values
        imarr = ImageArray(image)
        imarr = interpolateImage(imarr)
        return imarr

    @property
    def image(self):
        "Get image"
        return self.imarr.image

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
        return np.squeeze(intensity.reshape((self.imheight,
                                             self.imwidth, -1)))

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

    def normalize_surface_normal(self, surface_normal):
        "Normalize a given surface normal"
        ax1 = surface_normal[:, 0]
        ax2 = surface_normal[:, 1]
        ax3 = surface_normal[:, 2]
        normalized = np.zeros_like(surface_normal, dtype=np.float32)
        ax1norm = ax1 / np.linalg.norm(ax1)
        ax2norm = ax2 / np.linalg.norm(ax2)
        ax3norm = ax3 / np.linalg.norm(ax3)
        normalized[:, 0] = ax1norm
        normalized[:, 1] = ax2norm
        normalized[:, 2] = ax3norm
        return normalized

    def get_light_matrix(self, light_source: LightSource):
        "Get light vector from light source"
        coordarr = self.imarr.coordinates
        xarr = coordarr[:, 0]
        yarr = coordarr[:, 1]
        xdiff = light_source.x - xarr
        ydiff = light_source.y - yarr
        zval = light_source.z
        # TODO continue implementing light vector for dot product

    def change_diffuse_gain(self, g: float,
                            coeffarr: np.ndarray):
        """
        Change diffuse gain using the formula below:
        a_0' = g a_0
        a_1' = g a_1
        a_2' = g a_2
        a_3' = ( 1 − g )( 2 a_0 l_u_0 + a_2 l_v_0 ) + a_3
        a_4' = ( 1 − g )( 2 a_1 l_u_0 + a_2 l_v_0 ) + a_4
        a_5' = ( 1 − g )( a_0 l u_0_2 + a_1 l_v_0^2 + a_2 l_u_0 l_v_0 ) +
                ( a_3 − a_3 ' ' ) l_u_0 + ( a_4 − a_4 ' ) l_v_0 + a_5
        """
        # assert g > 0 and g < 1
        newarr = np.copy(coeffarr)
        newarr[:, 0] = coeffarr[:, 0] * g
        newarr[:, 1] = coeffarr[:, 1] * g
        newarr[:, 2] = coeffarr[:, 2] * g
        midterm = 2 * coeffarr[:, 0] * self.light_dirU_vec
        midterm2 = coeffarr[:, 2] * self.light_dirV_vec
        midterm += midterm2
        newarr[:, 3] = (1 - g) * midterm + coeffarr[:, 3]
        midterm = 2 * coeffarr[:, 1] * self.light_dirU_vec
        midterm += midterm2
        newarr[:, 4] = (1 - g) * midterm + coeffarr[:, 4]
        sum_first = coeffarr[:, 0] * self.light_dirU_vec**2
        sum_second = coeffarr[:, 1] * self.light_dirV_vec**2
        sum_third = coeffarr[:, 2] * (
            self.light_dirV_vec * self.light_dirU_vec
        )
        threeterm_sum = sum_first + sum_second + sum_third
        first_diff = (
            coeffarr[:, 3] - newarr[:, 3]) * self.light_dirU_vec
        second_diff = (
            coeffarr[:, 4] - newarr[:, 4]) * self.light_dirV_vec
        first = (1 - g) * threeterm_sum + first_diff + second_diff
        newarr[:, 5] = first + coeffarr[:, 5]

        return newarr

    def render_diffuse_gain(self, coeffarr: np.ndarray):
        "render change in diffuse gain"
        newcoeff = self.change_diffuse_gain(self.g, coeffarr)
        image = self.computeImage(newcoeff)
        return image

    def multiply_normal_luminance(self, normal, luminance):
        "Multiply the normal with luminance"
        # Dot product requires a shape match since we don't work
        # with pixels but coefficient arrays its easier and cheaper
        # to work with vector multiplications
        assert luminance.ndim == 1
        assert normal.ndim == 2
        assert normal.shape[1] == 3
        assert luminance.shape[0] == normal.shape[0]
        ax0 = normal[:, 0] * luminance
        ax1 = normal[:, 1] * luminance
        ax2 = normal[:, 2] * luminance
        newnorm = np.array([ax0, ax1, ax2])
        newnorm = newnorm.T
        newnorm = newnorm.sum(axis=1)
        return newnorm

    def compute_shading(self, coeffarr: np.ndarray):
        """
        Implementing the following equation for shading

        .. math::

            I_{\lambda} = I_{a\lambda} k_a O_{d\lambda}
                  + f_{att} I_{p\lambda} [k_{d}O_{d\lambda} (N \cdot L)
                  + k_{s}O_{s\lambda} (N \cdot R)^n]

        I_a :  ambient light for channel lambda
        k_a : coefficient for the ambient light for channel lambda
        f_att : light source attenuation function:
        f_att = min(\frac{1}{c_1 + c_2{\times}d_L + c_3{\times}d^2_{L}} , 1)
        c1, c2, c3 are provided by the user. d is the distance to light source
        I_p : light source primary component intensity
        k_d : coefficient for object diffusion color
        O_d : object diffusion color for channel lambda
        N: surface normal
        L: Light source position
        k_s : coefficient of object's specular color
        O_s : object's specular color
        R: is the reflection vector which is equal to
        R = 2 * N ( N.dot(L) ) - L
        n: specular component
        """
        assert self.ka > 0 and self.ka <= 1
        assert self.kd > 0 and self.kd <= 1
        first_multi = self.Ia * self.ka * self.od
        fatt = min(1/(self.c1 + self.c2 * self.dL + self.c3 * self.dL**2), 1)
        normal = self.get_surface_normal(coeffarr)
        luminance = self.get_luminance(coeffarr)
        # pdb.set_trace()
        dprod = self.multiply_normal_luminance(normal, luminance)
        second_multi = fatt * self.Ip * self.kd * self.od * dprod
        ref_vec = (2 * self.multiply_normal_luminance(normal, dprod)
                   ) - luminance
        third_multi = (
            self.multiply_normal_luminance(normal, ref_vec) ** self.n
        ) * self.ks * self.Os
        result = first_multi + second_multi + third_multi
        return result

    def shade_with_light_source(self, rgb_image: np.ndarray,
                                angle: int, elevation: int, cmap_fn=None):
        "Continue"
        # TODO matplotlib has a shading and light source algorithm
        # use their shade_rgb directly
        assert angle > 0 and angle <= 360
        assert elevation > 0 and elevation < 90
        grayim = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        light = Lsource(angle, elevation)
        # pdb.set_trace()
        if cmap_fn is not None:
            newimage = light.shade(data=grayim, cmap=cmap_fn)
        else:
            newimage = light.shade(data=grayim, cmap=plt.cm.hsv)
        shape = newimage.shape
        newimage = newimage.flatten()
        newimage = np.uint8(np.interp(newimage,
                                      (newimage.min(),
                                       newimage.max()),
                                      (0, 255))
                            )
        newimage = newimage.reshape(shape)

        return newimage

def setUpHandler(ptmpath: str):
    "From parse ptm file from path and setup ptm handler"
    parser = PTMFileParse(ptmpath)
    out = parser.parse()
    handler = PTMHandler(
        coeffarr=out['coeffarr'],
        image_height=out['image_height'],
        image_width=out['image_width'],
        scales=out['scales'],
        biases=out['biases'])
    # handler.setUp()
    return handler
