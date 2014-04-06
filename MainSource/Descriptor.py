'''
Created on Apr 5, 2014

@author: yulu
'''
import vlfeat as vl
import numpy as np
import os.path
from RandomKeypoint import Keypoint
from PIL import Image

class Descriptor:       
    def generate_desc(self, img_data, kpts):
        '''
        cannot use opencv functions, use vl functions to 
        do this for us
        '''
        A, self.desc = vl.vl_sift(img_data, frames = kpts, orientations = True)
        
    def save_desc(self, file_dir, file_name):
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
        
        try:
            with open(os.path.join(file_dir, file_name), 'wb') as file_data: 
                np.save(file_data, self.desc)
        except IOError as ioerr:
            print ioerr
        
    def load_desc(self, file_dir, file_name):
        try:
            with open(os.path(file_dir, file_name), 'rb') as file_data:
                np.load(file_data, self.desc)
        
        except IOError as ioerr:
            print ioerr

#test the class
if __name__ == '__main__':
    
    #load keypoint
    kpt = Keypoint()  
    kpt.load_keypoint('./Keypoint/', 'keypoint_1000')
    
    #load image(load image, convert to np array with float type)
    img = Image.open("./Image/stand-image_01.jpg")
    img_data = np.asarray(img, dtype=float)
    
    #generate desc
    desc = Descriptor()
    desc.generate_desc(img_data, kpt.kpt)
    
    print desc.desc
    