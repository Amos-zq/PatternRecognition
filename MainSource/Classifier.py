'''
Created on Apr 6, 2014

@author: yulu
'''
import numpy as np
import vlfeat as vl

from Keypoint import Keypoint
from Descriptor import Descriptor
from Signature import Signature
from Tree import Tree
from Weight import Weight


        


def dist(data_sign, query_sign):
    diff = data_sign-query_sign
    diff = np.fabs(diff)
    return np.sum(diff)

if __name__ == '__main__':

    #Load trained database
    wt = Weight(180, 11110)
    wt.load_weights('./Signature/1000/', 'weights')
    wt.load_weighted_sign('./Signature/1000/', 'weighted_sign')
    
    #Load tree
    tr = vl._vlfeat.VlHIKMTree(0, 0)
    tr.load('./tree.vlhkm')
    '''match test
    '''
    for i in range(0, 180):
        desc_file_name = 'desc_'+str(i)
        sign_file_name = 'sign_'+str(i)
               
        #load keypoint
        desc = Descriptor()
        desc.load_desc('./Descriptor/', desc_file_name)

        #generate sign
        sign = Signature()
        s = sign.generate_sign(tr, desc.desc, 10, 4)
    
        #weight   
        weighted_sign = wt.weight_sign(s)
    
        #print weighted_sign[100:300]
        #print wt.weighted_sign[i, 100:300]   
        d=np.empty(180)
        for i in range(0, 180):   
            d[i] = dist(wt.weighted_sign[i,:], weighted_sign)
    
        perm = np.argsort(d)
        vote_for = np.floor((perm[0:3])/10)+1
    
        #print perm[0:5]
        #print vote_for                       
        votes = vl.vl_binsum(np.zeros(18), np.ones(3), vote_for)
        
        #print votes
        best = np.argmax(votes)
    
        print best    
        
        
