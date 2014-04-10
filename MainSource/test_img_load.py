'''
Created on Apr 10, 2014

@author: yulu
'''

import os.path
from PIL import Image, ImageOps
import numpy as np
from Descriptor import Descriptor
from Keypoint import Keypoint


if __name__ == '__main__':
    img = Image.open("./Image_large/ibis/image_0002.jpg").convert('L')
    [width, height] = img.size
    h = 480
    ratio = float(height)/h
    w = int(width/ratio)
    img = ImageOps.fit(img, [w, h] , Image.ANTIALIAS)
    
    img.save('./Image/test.png')
    
    print img
    

    img_data = np.asarray(img, dtype=float)

    kpt = Keypoint()
    kpt.generate_keypoint(1000, img.size[0], img.size[1], 1)

    desc = Descriptor()
    desc.generate_desc(img_data, kpt.kpt)
    
    print desc.desc
    print str(w)
