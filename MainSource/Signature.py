'''
Created on Apr 5, 2014

@author: yulu
'''

import numpy as np
import os.path
from Descriptor import Descriptor
import vlfeat as vl

class Signature:
    def generate_sign(self, A, K, depth):
        m, n = A.shape
        L = (K**(depth+1)-1) / (K-1) - 1
        
        self.sign = np.zeros(L)
        
        '''
        branch_offset                      | level_offset
        -----------------------------------+-------------
        s(0) K**0                          | 1
        s(0) K**1 + s(1)*K**0              | 1 + K
        s(0) K**2 + s(1)*K**1 + s(2)*K**0  | 1 + K + K**2
        ...                                | ...
        '''
        for i in range(0, n):
            level_offset = 0
            branch_offset = 0
            
            for j in range(0, m):
                s = A[j, i] - 1
                level_offset = K * level_offset + 1
                branch_offset = K * branch_offset + s
                
                self.sign[level_offset + branch_offset - 1] += 1
                
    def save_sign(self, file_dir, file_name):
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
            
        try:
            with open(os.path.join(file_dir, file_name), 'wb') as file_data:
                np.save(file_data, self.sign)
        except IOError as ioerror:
            print ioerror
    
    def load_sign(self, file_dir, file_name):
        try:
            with open(os.path.join(file_dir, file_name), 'rb') as file_data:
                self.sign = np.load(file_data)
        except IOError as ioerror:
            print ioerror        
                
if __name__=="__main__":

    
     
    for i in range(0, 180):
        #load keypoint
        desc = Descriptor()
        desc.load_desc('./Descriptor/', 'desc_'+str(i))
    
        #push down to the tree
        #Load tree
        tr = vl._vlfeat.VlHIKMTree(0, 0)
        tr.load('./tree.vlhkm')
        
        At = vl.vl_hikmeanspush(tr, desc.desc)
        
        sign = Signature() 
        k = i*1000  
        sign.generate_sign(At, 10, 4)
    
        sign.save_sign('./Signature/1000/', 'sign_'+str(i))
        print 'generate sign for desc ' + str(i)
        print sign.sign       
                
                
        