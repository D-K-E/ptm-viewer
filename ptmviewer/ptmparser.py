# author: Kaan Eraslan
# license: see, LICENSE
# No warranties, see LICENSE
import numpy as np


class PTMFileParse:
    """
    Parse .ptm files according to
    http://www.hpl.hp.com/research/ptm/downloads/PtmFormat12.pdf
    """

    def __init__(self, ptmpath: str):
        self.path = ptmpath
        with open(self.path, "rb") as f:
            self.raw = f.readlines()
            self.raw = [raw_line for raw_line in self.raw if raw_line]
        self.header = self.raw[0].decode("utf-8").strip()
        self.format = self.raw[1].decode("utf-8").strip()
        if self.format != "PTM_FORMAT_RGB":
            raise ValueError(
                "ptm format {0} not supported".format(self.format)
            )

    @property
    def image_width(self):
        return int(self.raw[2].decode("utf-8").strip())

    @property
    def image_height(self):
        return int(self.raw[3].decode("utf-8").strip())

    @property
    def scales(self):
        scales = self.raw[4].decode("utf-8").strip().split()
        return np.array([float(s) for s in scales], dtype=np.float32)

    @property
    def biases(self):
        biases = self.raw[5].decode("utf-8").strip().split()
        return np.array([int(b) for b in biases], dtype=np.int32)

    def get_coeffarr(self):
        "Get coefficients array from bytelist"
        if self.format == "PTM_FORMAT_RGB":
            bytelist = self.raw[6:]
            # bytelist = reversed(bytelist)
            bstr = b"".join(bytelist)
            bstr = bstr[::-1]  # reverses the byte string due to format
            flatarr = np.frombuffer(bstr, dtype=np.uint8)
            flatarr = flatarr.reshape((-1, 6))
            flatarr = self.get_final_coefficient(flatarr)
        else:
            raise ValueError(
                """
                Working with an unsupported format {0}.
                Only uncompressed PTM_FORMAT_RGB is supported
                """.format(
                    self.format
                )
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
        out["coeffarr"] = coeffarr
        out["scales"] = self.scales
        out["biases"] = self.biases
        out["image_width"] = self.image_width
        out["image_height"] = self.image_height
        out["format"] = self.format
        return out
