'''
Created on Apr 6, 2014

@author: yulu
'''

import vlfeat as vl
import numpy as np
from Descriptor import Descriptor
import os.path

class Tree:      
    def generate_tree(self, data, K, nleaves):
        
        self.K = K
        self.nleaves = nleaves
        [tr, A] = vl.vl_hikmeans(data, self.K, self.nleaves, verb=1)
        
        print A[:, 0:1000]
            
        tr.save('./tree.vlhkm')
        
        #save A in signature
        #try:
        #    with open(os.path.join('./Signature/', 'sign_1000'), 'wb') as file_data:
        #        np.save(file_data, A)
        #except IOError as ioerror:
        #    print ioerror
        
if __name__=="__main__":
    desc = Descriptor()
    data = np.empty(shape=(128, 180*1000))
    k = 0
    for i in range(0, 180):
        desc.load_desc('./Descriptor/', 'desc_'+str(i))
        data[:, k:k+1000] = desc.desc
        print 'desc_'+str(i)
        k += 1000
        
    tree = Tree()
    tree.generate_tree(data, 10, 10000)
    
    #tr = vl._vlfeat.VlHIKMTree(0, 0)
    #tr.load('./tree.vlhkm')

    #At = vl.vl_hikmeanspush(tr, data[:, 0:1000])
    
    #print At[:, 0:1000]
    
    print data[:, 0:1000]
    
    