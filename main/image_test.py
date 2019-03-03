import core
import numpy as np
from PIL import Image

path = './assets/ptms/PF_Y_OB1_2750.ptm'
handler = core.setUpHandler(path)

def main():
    ""
    strinput = input('enter test savename: ')
    img = handler.ptm.image
    Image.fromarray(img).save(strinput)

if __name__ == '__main__':
    main()
