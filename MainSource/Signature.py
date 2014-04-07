'''
Created on Apr 5, 2014

Signature class takes in paramaters:
    tree
    desc
    K
    depth

@author: yulu
'''

import numpy as np
import os.path
from Descriptor import Descriptor
import vlfeat as vl

class Signature:
    def generate_sign(self, tr, desc, K, depth):
        L = (K**(depth+1)-1) / (K-1) - 1
        
        A = vl.vl_hikmeanspush(tr, desc)
        
        m, n = A.shape
       
        sign = np.zeros(L)
        
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
                
                sign[level_offset + branch_offset - 1] += 1
                
        return sign
                
    def generate_sign_database(self, tr, decs_file_dir, K, depth, total_img):
        L = (K**(depth+1)-1) / (K-1) - 1
        self.sign_database = np.empty(shape=(total_img, L))
        
        for i in range(0, total_img):
            #load keypoint
            desc = Descriptor()
            desc.load_desc(decs_file_dir, 'desc_'+str(i))
            
            sign = self.generate_sign(tr, desc.desc,  K, depth)
            
            print sign[0:30]
            self.sign_database[i,:] = sign

    def generate_sign_database_dir(self, tr, desc_database, K, depth, total_img, num_kpts):
        L = (K**(depth+1)-1) / (K-1) - 1
        self.sign_database = np.empty(shape=(total_img, L))
        
        for i in range(0, total_img):           
            sign = self.generate_sign(tr, desc_database[:,i*num_kpts:,(i+1)*num_kpts],  K, depth)
            
            self.sign_database[i,:] = sign                       
                
    def save_sign(self, file_dir, file_name):
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
            
        try:
            with open(os.path.join(file_dir, file_name), 'wb') as file_data:
                np.save(file_data, self.sign_database)
        except IOError as ioerror:
            print ioerror
    
    def load_sign(self, file_dir, file_name):
        try:
            with open(os.path.join(file_dir, file_name), 'rb') as file_data:
                self.sign_database = np.load(file_data)
        except IOError as ioerror:
            print ioerror        
                
if __name__=="__main__":

    #load tree
    tr = vl._vlfeat.VlHIKMTree(0, 0)
    tr.load('./tree.vlhkm') 
    
    sign = Signature()
    sign.generate_sign_database(tr, './Descriptor/', 10, 4, 180)
    
    sign.save_sign('./Signature/1000/', 'sign_1000')
                
                
        