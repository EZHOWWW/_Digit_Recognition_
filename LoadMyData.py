import os
import pandas as pd
import numpy as np
from PIL import Image

def get_my_data(path_dir):
    path_dir = os.path.join(os.getcwd(), path_dir)
    result = []

    for filename in os.listdir(path_dir):
        print(filename)
        path_join = os.path.join(path_dir, filename)
        if os.path.isfile(path_join) and filename.endswith('.png'):
            cur_dig = filename[0]
            result.append(get_serie_from_im(Image.open(path_join), cur_dig))
    return pd.DataFrame(result)

def get_pixel_value(pix):
    return 1 - (sum(pix) / 3) / 255
def get_serie_from_im(im, cur_dig):
    pixels = [get_pixel_value(i) for i in im.getdata()]
    pixels.append(cur_dig)
    result = pd.Series(pixels)
    return result





