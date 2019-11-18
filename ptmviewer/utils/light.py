# Author: Kaan Eraslan
# purpose implements a light object


import math
from PySide2.QtGui import QVector3D
from PySide2.QtGui import QVector4D
from ptmviewer.utils.utils import normalize_tuple
from ptmviewer.utils.utils import vec2vecDot
from ptmviewer.utils.utils import crossProduct
from ptmviewer.utils.utils import computeFrontRightPure
from ptmviewer.utils.utils import computeFrontRightQt
from ptmviewer.utils.obj3d import PureRigid3dObject
from ptmviewer.utils.obj3d import QtRigid3dObject
from ptmviewer.utils.obj3d import AbstractRigid3dObject

from abc import ABC, abstractmethod


class AbstractLightSource(ABC):
    def __init__(self):
        self.intensity = {}
        self.attenuation = {}
        self.coeffs = {}
        self.color = {}
        self.cutOff = math.cos(math.radians(12.5))
        self.outerCutOff = math.cos(math.radians(15.5))

    @abstractmethod
    def set_attenuation(self, attenuation):
        raise NotImplementedError

    @abstractmethod
    def set_color(self):
        raise NotImplementedError

    def check_intensity_coeff(self, val: float, valname: str):
        if not isinstance(val, float):
            raise TypeError(
                "Given " + valname + " is not of type float: ", str(type(val))
            )
        if not (val >= 0.0 and val <= 1.0):
            raise ValueError(
                "value " + valname + " not in given range 0.0 - 1.0: " + str(val)
            )

    def check_notFloat_proc(self, val: float, name: str):
        if not isinstance(val, float):
            raise TypeError(name + " is not of type float: " + str(type(val)))

    @abstractmethod
    def set_channel_intensity(self, channel: str, val: float):
        raise NotImplementedError

    @abstractmethod
    def set_channel_coeff(self, channel: str, val: float):
        raise NotImplementedError

    def set_cut_off(self, val: float):
        ""
        self.check_notFloat_proc(val, "cut off")
        self.cutOff = math.cos(math.radians(val))

    def set_outer_cut_off(self, val: float):
        ""
        self.check_notFloat_proc(val, "outer cut off")
        self.outerCutOff = math.cos(math.radians(val))

    def get_coeff_average(self):
        "Get the average value of its coefficients"
        counter = 0
        for val in self.coeffs.values():
            counter += val
        return counter / len(self.coeffs)


class PureLightSource(AbstractLightSource):
    "A pure python light source implementation"

    def __init__(
        self,
        cutOff=12.5,
        outerCutOff=15.5,
        attenuation={"constant": 1.0, "linear": 0.7, "quadratic": 1.8},
        intensity={"r": 1.0, "g": 1.0, "b": 1.0},
        coeffs={"r": 1.0, "g": 1.0, "b": 1.0},
    ):
        ""
        self.intensity = intensity
        self.coeffs = coeffs
        self.color = {}
        self.set_color()
        self.cutOff = math.cos(math.radians(cutOff))
        self.outerCutOff = math.cos(math.radians(outerCutOff))
        self.attenuation = attenuation
        self.attenVals = [
            # data taken on 2019-08-30 from
            # https://learnopengl.com/Lighting/Light-casters
            # distance, attenConst, attenLin, attenQaud
            [7, 1.0, 0.14, 0.07],
            [13, 1.0, 0.35, 0.44],
            [20, 1.0, 0.22, 0.20],
            [32, 1.0, 0.14, 0.07],
            [50, 1.0, 0.09, 0.032],
            [65, 1.0, 0.07, 0.017],
            [100, 1.0, 0.045, 0.0075],
            [160, 1.0, 0.027, 0.0028],
            [200, 1.0, 0.022, 0.0019],
            [325, 1.0, 0.014, 0.0007],
            [600, 1.0, 0.007, 0.0002],
            [3250, 1.0, 0.0014, 0.000007],
        ]

    def setAttenuationByTableVals(self, index: int):
        "Set attenuation values by table"
        row = self.attenVals[index]
        self.attenuation["constant"] = row[1]
        self.attenuation["linear"] = row[2]
        self.attenuation["quadratic"] = row[3]

    def setAttenuationValuesByDistance(self, distance: float):
        ""
        self.attenVals.sort(key=lambda x: x[0])
        maxdist = self.attenVals[-1][0]
        mindist = self.attenVals[0][0]
        if distance >= maxdist:
            self.setAttenuationByTableVals(-1)
            return
        if distance <= mindist:
            self.setAttenuationByTableVals(0)
            return
        for i, attenlst in enumerate(self.attenVals):
            dist = attenlst[0]
            if dist > distance:
                self.setAttenuationByTableVals(i)
                return

    def computeAttenuation4Distance(self, distance: float):
        "compute attenuation value for given distance"
        second = self.attenuation["linear"] * distance
        third = self.attenuation["quadratic"] * distance * distance
        return min(1, 1 / (self.attenuation["constant"] + second + third))

    def set_attenuation(self, atten: dict):
        "set attenuation"
        constant = atten["constant"]
        linear = atten["linear"]
        quadratic = atten["quadratic"]
        self.check_notFloat_proc(constant, name="constant attenuation")
        self.check_notFloat_proc(linear, name="linear attenuation")
        self.check_notFloat_proc(quadratic, name="quadratic attenuation")
        self.attenuation["constant"] = constant
        self.attenuation["linear"] = linear
        self.attenuation["quadratic"] = quadratic

    def set_color(self):
        "Set color"
        self.color = {}
        for channel, val in self.intensity.items():
            coeff = self.coeffs[channel]
            self.color[channel] = coeff * val

    @staticmethod
    def set_channel_val(channel: str, val: float):
        channel_d = {}
        name = channel.lower()
        if name == "red" or name == "r":
            channel_d["r"] = val
        elif name == "green" or name == "g":
            channel_d["g"] = val
        elif name == "blue" or name == "b":
            channel_d["b"] = val
        elif name == "alpha" or name == "a":
            channel_d["a"] = val
        else:
            mess = "Unrecognized channel name " + name
            mess += ", available channels are: "
            mess += "red, green, blue, alpha"
            raise ValueError(mess)
        return channel_d

    def set_channel_coeff(self, channel: str, val: float):
        "Set coefficients"
        self.check_intensity_coeff(val=val, valname=channel + " coefficient")
        #
        name = channel.lower()
        cdict = PureLightSource.set_channel_val(channel, val)
        self.coeffs.update(cdict)
        self.set_color()

    def set_channel_intensity(self, channel: str, val: float):
        "set channel intensity"
        self.check_intensity_coeff(val, channel + " intensity")
        #
        name = channel.lower()
        cdict = PureLightSource.set_channel_val(channel, val)
        self.intensity.update(cdict)
        self.set_color()

    def __str__(self):
        ""
        mess = "Light Source:\n position {0},\n direction {1},\n intensity {2}"
        mess += ",\n coefficients {3},\n color {4},\n cutOff value {5}"
        mess += ",\n outerCutOff value {9}, attenuation constant {6},"
        mess += "\n attenuation linear {7}"
        mess += ",\n attenuation quadratic {8}"
        return mess.format(
            str(self.position),
            str(self.direction),
            str(self.intensity),
            str(self.coeffs),
            str(self.color),
            str(self.cutOff),
            str(self.attenConst),
            str(self.attenLinear),
            str(self.attenQuad),
            str(self.outerCutOff),
        )


class QtLightSource(AbstractLightSource):
    "A light source"

    def __init__(
        self,
        intensity=QVector4D(1.0, 1.0, 1.0, 1.0),
        coefficients=QVector4D(1.0, 1.0, 1.0, 1.0),
        attenuation=QVector3D(1.0, 0.14, 0.07),
        cutOff=12.5,
        outerCutOff=15.0,
    ):
        ""
        self.color = QVector4D()
        self.intensity = intensity
        self.coeffs = coefficients
        self.set_color()
        self.cutOff = math.cos(math.radians(cutOff))
        self.outerCutOff = math.cos(math.radians(outerCutOff))
        self.attenuation = attenuation
        self.attenVals = [
            # data taken on 2019-08-30 from
            # https://learnopengl.com/Lighting/Light-casters
            # distance, attenConst, attenLin, attenQaud
            [7, 1.0, 0.14, 0.07],
            [13, 1.0, 0.35, 0.44],
            [20, 1.0, 0.22, 0.20],
            [32, 1.0, 0.14, 0.07],
            [50, 1.0, 0.09, 0.032],
            [65, 1.0, 0.07, 0.017],
            [100, 1.0, 0.045, 0.0075],
            [160, 1.0, 0.027, 0.0028],
            [200, 1.0, 0.022, 0.0019],
            [325, 1.0, 0.014, 0.0007],
            [600, 1.0, 0.007, 0.0002],
            [3250, 1.0, 0.0014, 0.000007],
        ]

    def setAttenuationByTableVals(self, index: int):
        "Set attenuation values by table"
        row = self.attenVals[index]
        self.attenuation = QVector4D(row[1], row[2], row[3], 1.0)

    def setAttenuationByDistance(self, distance: float):
        ""
        self.attenVals.sort(key=lambda x: x[0])
        maxdist = self.attenVals[-1][0]
        mindist = self.attenVals[0][0]
        if distance >= maxdist:
            self.setAttenuationByTableVals(-1)
            return
        if distance <= mindist:
            self.setAttenuationByTableVals(0)
            return
        for i, (dist, aconst, alin, aquad) in enumerate(self.attenVals):
            if dist > distance:
                self.setAttenuationByTableVals(i)
                return

    def check_notFloat_vec_proc(self, vec, vecname: str):
        "check whether all members of vectors are float"
        tpl = vec.toTuple()
        for t in tpl:
            self.check_notFloat_proc(t, name=vecname + " member")

    def set_attenuation(self, atten):
        "set attenuation"
        self.check_notFloat_vec_proc(atten, vecname="attenuation vector")
        self.attenuation = atten

    def set_color(self):
        "Set light source color using coeffs and intensities"
        if isinstance(self.intensity, QVector3D):
            self.color = QVector3D(
                self.intensity.x() * self.coeffs.x(),
                self.intensity.y() * self.coeffs.y(),
                self.intensity.z() * self.coeffs.z(),
            )
        else:
            self.color = QVector4D(
                self.intensity.x() * self.coeffs.x(),
                self.intensity.y() * self.coeffs.y(),
                self.intensity.z() * self.coeffs.z(),
                self.intensity.w() * self.coeffs.w(),
            )
            self.color = self.color.toVector3DAffine()

    def set_channel_val(self, channel: str, val: float, isIntensity=False):
        cvec = self.intensity if isIntensity else self.coeffs
        name = channel.lower()
        if name == "red" or name == "r":
            cvec.setX(val)
        elif name == "green" or name == "g":
            cvec.setY(val)
        elif name == "blue" or name == "b":
            cvec.setZ(val)
        elif name == "alpha" or name == "a":
            cvec.setW(val)
        else:
            mess = "Unrecognized channel name " + name
            mess += ", available channels are: "
            mess += "red, green, blue, alpha"
            raise ValueError(mess)
        self.set_color()
        return

    def set_channel_coeff(self, channel: str, val: float):
        "Set coefficient to given intesity"
        self.set_channel_val(channel=channel, val=val, isIntensity=False)

    def set_channel_intensity(self, channel: str, val: float):
        "Set coefficient to given intesity"
        self.set_channel_val(channel=channel, val=val, isIntensity=True)

    def get_coeff_average(self):
        "get average value for coefficients"
        counter = 0
        tplsize = 0
        for el in self.coeffs.toTuple():
            tplsize += 1
            counter += el
        return counter / tplsize

    def fromPureLightSource(self, light: PureLightSource):
        ""
        self.set_channel_intensity(channel="r", val=light.intensity["r"])
        self.set_channel_intensity(channel="g", val=light.intensity["g"])
        self.set_channel_intensity(channel="b", val=light.intensity["b"])
        atten = QVector3D()
        atten.setX(light.attenuation["constant"])
        atten.setY(light.attenuation["linear"])
        atten.setZ(light.attenuation["quadratic"])
        self.set_attenuation(atten)
        self.set_channel_coeff(channel="r", val=light.coeffs["r"])
        self.set_channel_coeff(channel="g", val=light.coeffs["g"])
        self.set_channel_coeff(channel="b", val=light.coeffs["b"])
        self.cutOff = light.cutOff

    def toPureLightSource(self):
        ""
        light = PureLightSource()
        light.setCutOff(self.cutOff)
        atten = {}
        atten["constant"] = self.attenuation.x()
        atten["linear"] = self.attenuation.y()
        atten["quadratic"] = self.attenuation.z()
        light.setAttenuation(atten)
        light.set_channel_coeff(channel="r", val=self.coeffs.x())
        light.set_channel_coeff(channel="g", val=self.coeffs.y())
        light.set_channel_coeff(channel="b", val=self.coeffs.z())
        if isinstance(self.coeffs, QVector4D):
            light.set_channel_coeff(channel="a", val=self.coeffs.w())
        light.set_channel_intensity(channel="r", val=self.intensity.x())
        light.set_channel_intensity(channel="g", val=self.intensity.y())
        light.set_channel_intensity(channel="b", val=self.intensity.z())
        if isinstance(self.intensity, QVector4D):
            light.set_channel_intensity(channel="a", val=self.intensity.w())
        return light


class PureLambertianReflector:
    "Object that computes lambertian reflection"

    def __init__(
        self,
        lightSource: PureLightSource,
        objDiffuseReflectionCoefficientRed: float,
        objDiffuseReflectionCoefficientGreen: float,
        objDiffuseReflectionCoefficientBlue: float,
        surfaceNormal: (int, int, int),
    ):
        self.light = lightSource
        self.objR = objDiffuseReflectionCoefficientRed
        self.objG = objDiffuseReflectionCoefficientGreen
        self.objB = objDiffuseReflectionCoefficientBlue
        assert self.objR <= 1.0 and self.objR >= 0.0
        assert self.objG <= 1.0 and self.objG >= 0.0
        assert self.objB <= 1.0 and self.objB >= 0.0
        self.costheta = None
        assert len(surfaceNormal) == 3
        self.surfaceNormal = surfaceNormal
        self.setCosTheta()
        self.reflection = {}
        self.setLambertianReflection()

    def setCosTheta(self):
        ""
        lightDir = self.light.direction
        lightDir = (lightDir["x"], lightDir["y"], lightDir["z"])
        normLight = normalize_tuple(lightDir)
        normSurf = normalize_tuple(self.surfaceNormal)
        self.costheta = vec2vecDot(normSurf, normLight)

    def setLambertianReflection(self):
        "compute reflection"
        red = self.light.intensity["r"] * self.objR * self.costheta
        green = self.light.intensity["g"] * self.objG * self.costheta
        blue = self.light.intensity["b"] * self.objB * self.costheta
        self.reflection["r"] = red
        self.reflection["g"] = green
        self.reflection["b"] = blue


class PureLambertianReflectorAmbient(PureLambertianReflector):
    "Pure python implementation of lambertian reflector with ambient light"

    def __init__(
        self,
        lightSource: PureLightSource,
        ambientLight: PureLightSource,
        objDiffuseReflectionCoefficientRed: float,
        objDiffuseReflectionCoefficientGreen: float,
        objDiffuseReflectionCoefficientBlue: float,
        surfaceNormal: (int, int, int),
    ):
        super().__init__(
            lightSource,
            objDiffuseReflectionCoefficientRed,
            objDiffuseReflectionCoefficientGreen,
            objDiffuseReflectionCoefficientBlue,
            surfaceNormal,
        )
        self.setCosTheta()
        self.setLambertianReflection()
        self.ambientLight = ambientLight

    def setLambertianReflectionWithAmbient(self):
        red = self.reflection["r"] + self.ambientLight.color["r"]
        green = self.reflection["g"] + self.ambientLight.color["g"]
        blue = self.reflection["b"] + self.ambientLight.color["b"]
        self.reflection = {"r": red, "g": green, "b": blue}


class AbstractShaderLight:
    def __init__(self):
        self.attenuation = {"constant": 0.0, "linear": 0.0, "quadratic": 0.0}
        self.ambient = None
        self.diffuse = None
        self.specular = None
        self.availableLightSources = ["ambient", "diffuse", "specular"]
        self.cutOff = 0.0
        self.outerCutOff = 0.0

    def set_cut_off(self, val: float):
        "set cut off value to diffuse and specular"
        self.diffuse.set_cut_off(val)
        self.specular.set_cut_off(val)
        self.cutOff = self.specular.cutOff

    def set_attenuation(self, atten):
        self.diffuse.set_attenuation(atten)
        self.specular.set_attenuation(atten)
        self.attenuation = self.specular.attenuation

    def set_channel_intensity(self, channel: str, val: float, lsource="diffuse"):
        "channel intensity"
        if lsource not in self.availableLightSources:
            raise ValueError("Unavailable light source: " + lsource)
        if lsource == "diffuse":
            self.diffuse.set_channel_intensity(channel, val)
        elif lsource == "specular":
            self.specular.set_channel_intensity(channel, val)
        else:
            self.ambient.set_channel_intensity(channel, val)

    def set_channel_coeff(self, channel: str, val: float, lsource="diffuse"):
        "channel coefficient"
        if lsource not in self.availableLightSources:
            raise ValueError("Unavailable light source: " + lsource)
        if lsource == "diffuse":
            self.diffuse.set_channel_coeff(channel, val)
        elif lsource == "specular":
            self.specular.set_channel_coeff(channel, val)
        else:
            self.ambient.set_channel_coeff(channel, val)

    def set_outer_cut_off(self, val: float):
        "set cut off value to diffuse and specular"
        self.diffuse.set_outer_cut_off(val)
        self.specular.set_outer_cut_off(val)
        self.outerCutOff = self.specular.outerCutOff


class PureShaderLight(AbstractShaderLight, PureRigid3dObject):
    "A Pure python shader light object for illumination"

    def __init__(
        self,
        position: dict,
        cutOff=12.5,
        outerCutOff=15.0,
        attenuation={"constant": 1.0, "linear": 0.7, "quadratic": 1.8},
        ambient=PureLightSource(coeffs={"r": 0.3, "g": 0.3, "b": 0.3}),
        diffuse=PureLightSource(),
        specular=PureLightSource(),
    ):
        ""
        super().__init__()
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.set_position(position)
        self.set_attenuation(attenuation)
        self.set_cut_off(cutOff)
        self.set_outer_cut_off(outerCutOff)

    def get_specular_color(self):
        return {
            "r": self.specular.color["r"],
            "g": self.specular.color["g"],
            "b": self.specular.color["b"],
        }

    def get_ambient_color(self):
        return {
            "r": self.ambient.color["r"],
            "g": self.ambient.color["g"],
            "b": self.ambient.color["b"],
        }

    def get_diffuse_color(self):
        return {
            "r": self.diffuse.color["r"],
            "g": self.diffuse.color["g"],
            "b": self.diffuse.color["b"],
        }

    def __str__(self):
        "string representation"
        mess = "Shader Light:\n position {0},\n ambient {2}"
        mess += ",\n diffuse {3},\n specular {4},\n cut off {5}"
        mess += ",\n outerCutOff value {9}, attenuation values {6},"
        return mess.format(
            str(self.position),
            str(self.ambient),
            str(self.diffuse),
            str(self.specular),
            str(self.cutOff),
            str(self.outerCutOff),
            str(self.attenuation),
        )


class QtShaderLight(AbstractShaderLight, QtRigid3dObject):
    "Qt shader light object"

    def __init__(
        self,
        position=QVector3D(0.0, 1.0, 0.0),
        cutOff=12.5,
        outerCutOff=15.0,
        attenuation=QVector3D(1.0, 0.14, 1.8),
        ambient=QtLightSource(),
        diffuse=QtLightSource(),
        specular=QtLightSource(),
    ):
        ""
        QtRigid3dObject.__init__(self)
        AbstractShaderLight.__init__(self)

        # rigid 3d object
        self.set_position(position)

        # shader light
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.set_cut_off(cutOff)
        self.set_outer_cut_off(outerCutOff)
        self.set_attenuation(attenuation)

