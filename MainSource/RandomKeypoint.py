'''
Created on Apr 5, 2014

@author: yulu
'''

import numpy as np
import os.path

class Keypoint:
    def generate_keypoint(self, 
                          num_of_kpt, 
                          width, 
                          height, 
                          sigma = 1,
                          orientation = 0):
        self.kpt = np.empty(shape=(4, num_of_kpt))
        '''
        Generate Random Keypoint as the format
        [x, y, scale, orientation]
        '''
        rmin = 6*sigma
        rmax = min(width, height)/2
        
        r_range = range(rmin,rmax)
        p = [width*height - 2*(width+height)*k + 4*k**2 for k in r_range]
        p_array = np.array(p, dtype=float)
        p_array = p_array / np.sum(p_array)
        P = np.cumsum(p_array)
        P_list = P.tolist()
        P_list.pop()
        P_list.insert(0,0)
        
        for i in range(0,num_of_kpt):
            rn = np.random.rand()
            sel = [k for k in range(0, len(r_range)) if P_list[k] < rn]
            r = r_range[max(sel)]
            
            w = width - 2*r
            h = height - 2*r
            X = w*np.random.rand() + r
            Y = h*np.random.rand() + r
            sig = r / 6
            
            self.kpt[:, i] = [X, Y, sig, orientation]            
         
        
    def save_keypoint(self, file_dir, file_name):
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
            
        try:
            with open(os.path.join(file_dir, file_name), 'wb') as file_data:
                np.save(file_data, self.kpt)
                
        except IOError as ioerr:
            print ioerr
        
    def load_keypoint(self, file_dir, file_name):
        try:
            with open(os.path.join(file_dir, file_name), 'rb') as file_data:
                np.load(file_data, self.kpt)
                
        except IOError as ioerr:
            print ioerr        
        

#Test the code
if __name__ == '__main__':
    kp = Keypoint()
    
    kp.generate_keypoint(1000, 640, 480, 1)
    
    kp.save_keypoint('./Keypoint/', 'keypoint_1000')
    
    print kp.kpt