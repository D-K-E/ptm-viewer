# author: Kaan Eraslan
# license: see, LICENSE
# No Warranties, use at your own risk, see LICENSE

import numpy as np


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
        self.image_width = int(self.raw[2].decode('utf-8').strip())
        self.image_height = int(self.raw[3].decode('utf-8').strip())
        self.scales = self.raw[4].decode('utf-8').strip().split()
        self.scales = np.array([float(s) for s in self.scales], dtype=np.float)
        self.biases = self.raw[5].decode('utf-8').strip().split()
        self.biases = np.array([int(b) for b in self.biases], dtype=np.int32)
        if self.format != "PTM_FORMAT_RGB":
            raise ValueError('ptm format {0} not supported'.format(self.format)
                             )

    def get_coeffarr(self):
        "Get coefficients array from bytelist"
        if self.format == "PTM_FORMAT_RGB":
            bytelist = self.raw[6:]
            bstr = b''.join(bytelist)
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


class PTMHandler:
    """
    Implements methods for manipulating Polynomial Texture Mappings (PTMs)
    """

    def __init__(self,
                 coeffarr: np.ndarray,
                 image_height: int,
                 image_width: int,
                 scales: [float],
                 biases: [int]
                 ):
        self.arr = np.copy(coeffarr)
        self.scales = scales
        self.biases = biases
        self.imheight = image_height
        self.imwidth = image_width
        self.light_dirV_vec = None
        self.light_dirU_vec = None
        self.light_mat = None
        self.L = None
        self.surface_normal = None
        self.image = None

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

    def set_light_dirU_vec(self, coeffarr: np.ndarray):
        "Set light direction U vector from coeffarr"
        self.light_dirU_vec = self.get_light_dirU_vec(coeffarr)

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

    def set_light_dirV_vec(self, coeffarr):
        "Set light direction v vector from coeffarr"
        self.light_dirV_vec = self.get_light_dirV_vec(coeffarr)

    def set_light_direction_vec(self, coeffarr):
        "Set light vector with computed values for v and u"
        self.light_mat = self.get_light_direction_vector(coeffarr)

    def form_light_direction_mat(self, luvec, lvvec):
        "Form the light direction matrice"
        l_mat = np.array([luvec**2,
                          lvvec**2,
                          lvvec * luvec,
                          luvec,
                          lvvec,
                          np.ones_like(lvvec)], dtype=np.float16)
        return np.transpose(l_mat, (1, 0))

    def get_light_direction_vector(self, coeffarr: np.ndarray):
        "Set light direction vector from coeffarr"
        luvec = self.get_light_dirU_vec(coeffarr)
        lvvec = self.get_light_dirV_vec(coeffarr)
        return self.form_light_direction_mat(luvec, lvvec)

    def get_luminance(self, coeffarr):
        "get luminance from coeffarr to have images"
        light_mat = self.get_light_direction_vector(coeffarr)
        flatim = np.sum(coeffarr * light_mat, dtype=np.uint8, axis=1)
        return np.squeeze(flatim)

    def set_luminance(self, coeffarr):
        "Set L value from coeffarr"
        self.L = self.get_luminance(coeffarr)

    def form_surface_normal(self, luvec,
                            lvvec):
        "Form surface normal matrice"
        normal = np.array(
            [luvec,
             lvvec,
             np.sqrt(1 - luvec**2 - lvvec**2)
             ],
            dtype=np.float16)
        # return np.transpose(normal, (1, 0))
        return normal

    def get_surface_normal(self, coeffarr):
        """
        Surface normal
        """
        luvec = self.get_light_dirU_vec(coeffarr)
        lvvec = self.get_light_dirV_vec(coeffarr)
        return self.form_surface_normal(luvec, lvvec)

    def set_surface_normal(self, coeffarr):
        "Surface normal"
        luvec = self.light_dirU_vec
        lvvec = self.light_dirV_vec
        self.surface_normal = self.form_surface_normal(luvec, lvvec)

    def set_halfway_vector(self):
        "Set halfway vector H using normal and light vec"
        self.halfway_vec = (
            (self.surface_normal + self.light_mat) /
            (self.surface_normal)  # TODO continue the formula p. 731
        )

    def set_light(self, coeffarr):
        "Set light direction vectors and matrices"
        self.set_light_dirU_vec(coeffarr)
        self.set_light_dirV_vec(coeffarr)
        self.light_mat = self.form_light_direction_mat(self.light_dirU_vec,
                                                       self.light_dirV_vec)

    def set_image(self, coeffarr):
        "Set image from coeffarr"
        self.set_light(coeffarr)
        self.set_luminance(coeffarr)
        self.image = self.L.reshape((self.imheight,
                                     self.imwidth,
                                     3))

    def setUp(self):
        "Setup handler from given coefficient array"
        self.set_image(coeffarr=self.arr)
        self.set_surface_normal(self.arr)

    def computeImage(self, coeffarr):
        "Compute image from coefficient array"
        flatim = self.get_luminance(coeffarr)
        image = flatim.reshape((self.imheight,
                                self.imwidth,
                                3))

        return image

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
        assert g > 0 and g < 1
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
        first_multi = self.Ia * self.ka * self.od
        fatt = min(1/(self.c1 + self.c2 * self.dL + self.c3 * self.dL**2), 1)
        normal = self.get_surface_normal(coeffarr)
        luminance = self.get_luminance(coeffarr)
        dprod = normal.dot(luminance)
        second_multi = fatt * self.Ip * self.kd * self.od * dprod
        ref_vec = (2 * normal * (normal.dot(luminance))) - luminance
        third_multi = (normal.dot(ref_vec) ** self.n) * self.ks * self.Os
        result = first_multi + second_multi + third_multi
        return result

    def render_coeffarr(self, coeffarr: np.ndarray):
        "Render coefficient array with values in class"
        newarr = self.change_diffuse_gain(self.g, coeffarr)
        shaded = self.compute_shading(newarr)
        return shaded  # should return an image instead TODO


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
    handler.setUp()
    return handler
