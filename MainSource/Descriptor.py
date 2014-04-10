'''
Created on Apr 5, 2014

Descriptor class takes in paramaters:
    img_data
    kpts

@author: yulu
'''
import vlfeat as vl
import numpy as np
import os.path
from Keypoint import Keypoint
from PIL import Image

class Descriptor:       
    def generate_desc(self, img_data, kpts):
        '''
        cannot use opencv functions, use vl functions to 
        do this for us
        '''
        drop, self.desc = vl.vl_sift(img_data, kpts, orientations = False) #why cannot change to true for orientation????
        
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
            with open(os.path.join(file_dir, file_name), 'rb') as file_data:
                self.desc = np.load(file_data)
                self.desc = self.desc.astype(float)
        
        except IOError as ioerr:
            print ioerr

#test the class
if __name__ == '__main__':
    
    #load keypoint
    kpt = Keypoint()  
    kpt.load_keypoint('./Keypoint/', 'keypoint_1000')
    
    k = 0
    num_of_train = 10
    for folder_name in os.listdir('./Image'):
        count = 0;
        for file_name in os.listdir(os.path.join('./Image', folder_name)):
            if count >= 10:
                break
            #load image(load image, convert to np array with float type)
            img = Image.open(os.path.join('./Image', folder_name, file_name))
            img_data = np.asarray(img, dtype=float)
    
            #generate desc
            desc = Descriptor()
            desc.generate_desc(img_data, kpt.kpt)
            desc.save_desc('./Descriptor/', 'desc_'+str(k))
            
            count += 1
            k += 1
            
            print desc.desc
    