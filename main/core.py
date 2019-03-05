# author: Kaan Eraslan
# license: see, LICENSE
# No Warranties, use at your own risk, see LICENSE

import numpy as np
# import cupy as np
import pdb

# from stack overflow
# https://stackoverflow.com/a/21096605/7330813


def _block_slices(dim_size, block_size):
    """Generator that yields slice objects for indexing into 
    sequential blocks of an array along a particular axis
    """
    count = 0
    while True:
        yield slice(count, count + block_size, 1)
        count += block_size
        if count > dim_size:
            raise StopIteration


def blockwise_dot(A, B, max_elements=int(2**27), out=None):
    """
    Computes the dot product of two matrices in a block-wise fashion. 
    Only blocks of `A` with a maximum size of `max_elements` will be 
    processed simultaneously.
    """

    m,  n = A.shape
    n1, o = B.shape

    if n1 != n:
        raise ValueError('matrices are not aligned')

    if A.flags.f_contiguous:
        # prioritize processing as many columns of A as possible
        max_cols = max(1, max_elements / m)
        max_rows = max_elements / max_cols

    else:
        # prioritize processing as many rows of A as possible
        max_rows = max(1, max_elements / n)
        max_cols = max_elements / max_rows

    if out is None:
        out = np.empty((m, o), dtype=np.result_type(A, B))
    elif out.shape != (m, o):
        raise ValueError('output array has incorrect dimensions')

    for mm in _block_slices(m, max_rows):
        out[mm, :] = 0
        for nn in _block_slices(n, max_cols):
            A_block = A[mm, nn].copy()  # copy to force a read
            out[mm, :] += np.dot(A_block, B[nn, :])
            del A_block

    return out

# end stack overflow


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


class LightSource:
    "Simple implementation of a light source"

    def __init__(self,
                 x=1.0,  # x
                 y=1.0,  # y
                 z=20.0,  # light source distance: 0 to make it at infinity
                 intensity=1.0,  # I_p
                 ambient_intensity=1.0,  # I_a
                 ambient_coefficient=0.000000002,  # k_a
                 ):
        "light source"
        self.x = x
        self.y = y
        if z is not None:
            assert isinstance(z, float)
        self.z = z
        self.intensity = intensity
        self.ambient_intensity = ambient_intensity  # I_a
        self.ambient_coefficient = ambient_coefficient  # k_a
        # k_a can be tuned if the material is known

    def copy(self):
        "copy self"
        return LightSource(x=self.x,
                           y=self.y,
                           z=self.z,
                           intensity=self.intensity,
                           light_power=self.power)


class ChannelShader:
    "Shades channels"

    def __init__(self,
                 coordarr: np.ndarray,
                 light_source: LightSource,  # has I_a, I_p, k_a
                 surface_normal: np.ndarray,
                 imagesize: (int, int),
                 color: np.ndarray,  # they are assumed to be O_d and O_s
                 spec_coeff=0.99999999,  # k_s
                 screen_gamma=2.2,
                 diffuse_coeff=0.1,  # k_d
                 attenuation_c1=1.0,  # f_attr c1
                 attenuation_c2=1.0,  # f_attr c2 d_L coefficient
                 attenuation_c3=0.0,  # f_attr c3 d_L^2 coefficient
                 shininess=20.0  # n
                 ):
        self.light_source = light_source
        self.light_intensity = self.light_source.intensity  # I_p
        self.ambient_coefficient = self.light_source.ambient_coefficient  # k_a
        self.ambient_intensity = self.light_source.ambient_intensity  # I_a
        self.coordarr = coordarr
        self.surface_normal = np.copy(surface_normal)
        self.screen_gamma = screen_gamma
        self.shininess = shininess
        self.diffuse_coeff = diffuse_coeff  # k_d
        self.diffuse_color = normalize_1d_array(color)  # O_d: obj diffuse color
        # self.diffuse_color = color  # O_d: obj diffuse color
        self.spec_color = normalize_1d_array(color)  # O_s: obj specular color
        self.spec_coeff = spec_coeff  # k_s: specular coefficient
        self.imsize = imagesize
        self.att_c1 = attenuation_c1
        self.att_c2 = attenuation_c2
        self.att_c3 = attenuation_c3

    def copy(self):
        return ChannelShader(coordarr=np.copy(self.coordarr),
                             light_source=self.light_source.copy(),
                             surface_normal=np.copy(self.surface_normal),
                             color=np.copy(self.diffuse_coeff) * 255.0)

    @property
    def distance(self):
        yarr = self.coordarr[:, 0]  # row nb
        xarr = self.coordarr[:, 1]  # col nb
        xdist = (self.light_source.x - xarr)**2
        ydist = (self.light_source.y - yarr)**2
        return xdist + ydist

    @property
    def distance_factor(self):
        resx = self.imsize[1]
        factor = self.distance / self.light_source.z * resx
        return 1.0 - factor

    @property
    def light_direction(self):
        "get light direction matrix (-1, 3)"
        yarr = self.coordarr[:, 0]
        xarr = self.coordarr[:, 1]
        xdiff = self.light_source.x - xarr
        ydiff = self.light_source.y - yarr
        light_matrix = np.zeros((self.coordarr.shape[0], 3))
        light_matrix[:, 0] = ydiff
        light_matrix[:, 1] = xdiff
        light_matrix[:, 2] = self.light_source.z
        # light_matrix[:, 2] = 0.0
        # pdb.set_trace()
        return light_matrix

    @property
    def light_attenuation(self):
        """
        Implementing from Foley JD 1996, p. 726

        f_att : light source attenuation function:
        f_att = min(\frac{1}{c_1 + c_2{\times}d_L + c_3{\times}d^2_{L}} , 1)
        """
        second = self.att_c2 * self.distance
        third = self.att_c3 * self.distance * self.distance
        result = self.att_c1 + second + third
        result = 1 / result
        return np.where(result < 1, result, 1)

    @property
    def normalized_light_direction(self):
        "Light Direction matrix normalized"
        return normalize_3col_array(self.light_direction)

    @property
    def normalized_surface_normal(self):
        return normalize_3col_array(self.surface_normal)

    @property
    def costheta(self):
        "set costheta"
        # pdb.set_trace()
        costheta = get_vector_dot(
            arr1=self.normalized_light_direction,
            arr2=self.normalized_surface_normal)
        # products of vectors
        # costheta = np.abs(costheta)  # as per (Foley J.D, et.al. 1996, p. 724)
        costheta = np.where(costheta > 0, costheta, 0)
        return costheta

    @property
    def ambient_term(self):
        "Get the ambient term I_a * k_a * O_d"
        term = self.ambient_coefficient * self.ambient_intensity
        term *= self.diffuse_color
        # pdb.set_trace()
        return term

    @property
    def view_direction(self):
        "Get view direction"
        # pdb.set_trace()
        cshape = self.coordarr.shape
        coord = np.zeros((cshape[0], 3))  # x, y, z
        coord[:, :2] = -self.coordarr
        coord[:, 2] = 0.0  # viewer at infinity
        coord = normalize_3col_array(coord)
        return coord

    @property
    def half_direction(self):
        "get half direction"
        # pdb.set_trace()
        arr = self.view_direction + self.normalized_light_direction
        return normalize_3col_array(arr)

    @property
    def spec_angle(self):
        "get spec angle"
        specAngle = get_vector_dot(
            arr1=self.half_direction,
            arr2=self.normalized_surface_normal)
        return np.where(specAngle > 0.0, specAngle, 0.0)

    @property
    def specular(self):
        return self.spec_angle ** self.shininess

    @property
    def channel_color_blinn_phong(self):
        """compute new channel color intensities
        Implements: Foley J.D. 1996 p. 730 - 731, variation on equation 16.15
        """
        second = 1.0  # added for structuring code in this fashion, makes
        # debugging easier
        # lambertian terms
        second *= self.diffuse_coeff  # k_d
        second *= self.costheta  # (N \cdot L)
        second *= self.light_intensity  # I_p
        # adding phong terms
        second *= self.light_attenuation  # f_attr
        second *= self.diffuse_color  # O_d
        third = 1.0
        third *= self.spec_color  # O_s
        third *= self.specular  # (N \cdot H)^n
        third *= self.spec_coeff  # k_s
        result = 0.0
        #
        result += self.ambient_term  # I_a × k_a × O_d
        result += second
        result += third
        # pdb.set_trace()
        return result


class Shader:
    "Shader regroups shading methods operating on pixels"

    def __init__(self,
                 red_channel_shader: ChannelShader,
                 green_channel_shader: ChannelShader,
                 blue_channel_shader: ChannelShader,
                 ):
        self.red_shader = red_channel_shader
        self.green_shader = green_channel_shader
        self.blue_shader = blue_channel_shader

    def copy(self):
        "copy self"
        return Shader(self.red_shader.copy(),
                      self.green_shader.copy(),
                      self.blue_shader.copy())

    def gamma_correct_color(self, color: np.ndarray):
        "Correct the gamma of the color array"
        return color ** (1.0 / self.screen_gamma)

    def shade_cell(self, surface_normals: np.ndarray):
        "Shade using cell shader"
        return None

    def shade_blinn_phong(self):
        "get blinn phong shading"
        shaded = np.array([self.red_shader.channel_color_blinn_phong,
                           self.green_shader.channel_color_blinn_phong,
                           self.blue_shader.channel_color_blinn_phong])
        return shaded.T

    def shade_normals(self, shadefn='phong'):
        "Shade using surface normals with specified function"
        shadefn = shadefn.lower()
        if shadefn != 'phong':
            raise ValueError(
                "Unsupported shader {0}, 'phong' is available".format(shadefn)
            )
        else:
            return self.shade_blinn_phong()


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


class PTMHandler:
    """
    Implements methods for manipulating Polynomial Texture Mappings (PTMs)
    """

    def __init__(self,
                 ptm: RGBPTM,
                 shader: Shader = None,
                 light_source: LightSource = None
                 ):
        self.shader = shader
        self.light_source = light_source
        self.ptm = ptm
        # see change diffuse gain for variable name
        self.g = 0.4

    def copy(self):
        "copy self"
        return PTMHandler(ptm=self.ptm.copy(),
                          shader=self.shader.copy(),
                          light_source=self.light_source.copy())

    def shade_ptm(self):
        "Shade ptm using shader"
        imshape = self.ptm.image.shape
        shaded = self.shader.shade_normals()
        # pdb.set_trace()
        shaded = shaded.reshape(imshape)
        imarr = ImageArray(shaded)
        imarr = interpolateImage(imarr)
        return np.fliplr(imarr.image)

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

    def compute_shading(self, coeffarr: np.ndarray):
        """
        Implementing the following equation for shading
        DEPECRATED, see Shader object

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
        pass


def setUpHandler(ptmpath: str):
    "From parse ptm file from path and setup ptm handler"
    parser = PTMFileParse(ptmpath)
    out = parser.parse()
    ptm = RGBPTM(
        coeffarr=out['coeffarr'],
        image_height=out['image_height'],
        image_width=out['image_width'],
        scales=out['scales'],
        biases=out['biases'])
    light_source = LightSource(x=float(out['image_width']),
                               y=float(out['image_height']),
                               ambient_coefficient=0.000000002,  # k_a
                               )
    coordarr = ptm.imarr.coordinates
    # pdb.set_trace()
    red_shader = ChannelShader(coordarr,
                               light_source,
                               imagesize=ptm.imarr.image.shape[:2],
                               color=ptm.red_channel_normalized_pixel_values,
                               surface_normal=ptm.red_channel_surface_normal)
    green_shader = ChannelShader(coordarr,
                                 light_source,
                                 imagesize=ptm.imarr.image.shape[:2],
                                 color=ptm.green_channel_normalized_pixel_values,
                                 surface_normal=ptm.green_channel_surface_normal)
    blue_shader = ChannelShader(coordarr,
                                light_source,
                                imagesize=ptm.imarr.image.shape[:2],
                                color=ptm.blue_channel_normalized_pixel_values,
                                surface_normal=ptm.blue_channel_surface_normal)
    shader = Shader(red_shader, green_shader, blue_shader)
    handler = PTMHandler(ptm, shader, light_source)
    return handler
