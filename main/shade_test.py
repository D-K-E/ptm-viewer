import core
import numpy as np
from PIL import Image
import subprocess

path = './assets/ptms/PF_Y_OB1_2750.ptm'
handler = core.setUpHandler(path)


def main():
    ""
    strinput = input('enter test savename: ')
    shaded_image = handler.shade_ptm()
    Image.fromarray(shaded_image).save(strinput)
    subprocess.run(['git', 'add', '-A'])
    params = {}
    params['light_source'] = {'direction': {
        'x': handler.light_source.x,
        'y': handler.light_source.y,
        'z': handler.light_source.z},
        'intensity': handler.light_source.intensity,
        'ambient_coefficient': handler.light_source.ambient_coefficient,
        'ambient_intensity': handler.light_source.ambient_intensity}
    spec_normed = False
    spec_color = handler.shader.blue_shader.spec_color
    if not isinstance(spec_color, float):
        spec_color = spec_color < 1
        spec_color = spec_color.all()
        if spec_color:
            spec_normed = True
    else:
        spec_normed = spec_color
    diff_normed = False
    diff_color = handler.shader.blue_shader.diffuse_color
    diff_color = diff_color < 1
    diff_color = diff_color.all()
    if diff_color:
        diff_normed = True
    params['shader'] = {
        'diffuse_coeff': handler.shader.blue_shader.diffuse_coeff,
        'spec_coeff': handler.shader.blue_shader.spec_coeff,
        'shininess': handler.shader.blue_shader.shininess,
        'f_attr_c1': handler.shader.blue_shader.att_c1,
        'f_attr_c2': handler.shader.blue_shader.att_c2,
        'f_attr_c3': handler.shader.blue_shader.att_c3,
        'diffuse_color_normalized': diff_normed,
        'spec_color_normalized': spec_normed
    }
    params['output_name'] = strinput
    strparams = ""
    for key, val in params.items():
        strparams += key + ' : ' + str(val)
        strparams += '\n'
    commit_message = """'Notes and Parameters

Here are the values of the parameters involved:

{0}'
""".format(strparams)
    subprocess.run(['git', 'commit',
                    '-m',
                    commit_message])

if __name__ == '__main__':
    main()
