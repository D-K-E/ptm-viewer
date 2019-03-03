import core
import numpy as np
from PIL import Image

path = './assets/ptms/PF_Y_OB1_2750.ptm'
handler = core.setUpHandler(path)

def main():
    ""
    strinput = input('enter test savename: ')
    shaded_image = handler.shade_ptm()
    Image.fromarray(shaded_image).save(strinput)

if __name__ == '__main__':
    main()
