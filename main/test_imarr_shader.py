import core
import numpy as np
import arrayfire as af
from PIL import Image
import subprocess

path = './assets/ptms/PF_Y_OB1_2750.ptm'


def shade_image(path, backend="arrayfire"):
    parser = core.PTMFileParse(path)
    out = parser.parse()
    ptm = core.RGBPTM(
        coeffarr=out['coeffarr'],
        image_height=out['image_height'],
        image_width=out['image_width'],
        scales=out['scales'],
        biases=out['biases'])
    light_source = core.LightSource(x=float(out['image_width']),
                                    y=float(out['image_height']),
                                    ambient_coefficient=0.000000002,  # k_a
                                    )

    imarr = ptm.imarr
    imarr.setBackend(backend)

    shader = core.ShaderImArray(imarr,
                                light_source,
                                surface_normal=ptm.surface_normals)
    shaded = shader.color_blinn_phong
    ashape = imarr.arrshape

    if imarr.backend == "numpy":
        shaded = shaded.reshape(ashape)
    elif imarr.backend == "arrayfire":
        shaded = af.moddims(shaded, *ashape)

    shaded = core.ImageArray(shaded)
    shaded = core.interpolateImage(shaded)
    return shaded, light_source, shader


def main():
    ""
    strinput = input('enter test savename: ')
    strinput2 = input('backend [N: numpy / a: arrayfire] : ')
    if strinput2 == "a":
        backend = "arrayfire"
    else:
        backend = "numpy"
    imarr, light_source, shader = shade_image(path, backend)
    if backend == 'numpy':
        Image.fromarray(imarr.image).save(strinput)
    elif backend == 'arrayfire':
        Image.fromarray(imarr.image.to_ndarray().astype('uint8')
                        ).save(strinput)
    subprocess.run(['git', 'add', '-A'])
    params = {}
    params['light_source'] = {'direction': {
        'x': light_source.x,
        'y': light_source.y,
        'z': light_source.z},
        'intensity': light_source.intensity,
        'ambient_coefficient': light_source.ambient_coefficient,
        'ambient_intensity': light_source.ambient_intensity}
    spec_normed = False
    spec_color = shader.spec_color
    if not isinstance(spec_color, float):
        spec_color = spec_color < 1
        spec_color = spec_color.all()
        if spec_color:
            spec_normed = True
    else:
        spec_normed = spec_color
    diff_normed = False
    diff_color = shader.diffuse_color
    diff_color = diff_color < 1
    diff_color = diff_color.all()
    if diff_color:
        diff_normed = True
    params['shader'] = {
        'diffuse_coeff': shader.diffuse_coeff,
        'spec_coeff': shader.spec_coeff,
        'shininess': shader.shininess,
        'f_attr_c1': shader.att_c1,
        'f_attr_c2': shader.att_c2,
        'f_attr_c3': shader.att_c3,
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
