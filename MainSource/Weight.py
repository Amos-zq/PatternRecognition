'''
Created on Apr 5, 2014

@author: yulu
'''
import numpy as np
from Signature import Signature
import os.path

class Weight:
    def __init__(self, cutoff, total_img, K, depth):
        L = (K**(depth+1)-1) / (K-1) - 1
        self.weighted_sign = np.empty(shape=(total_img, L))
        self.cutoff = cutoff
        
        # load signature
        for i in range(0, total_img):
            sign = Signature()
            sign.load_sign('./Signature/1000/', 'sign_'+str(i))
            self.weighted_sign[i, :] = sign.sign
            print 'load sign_'+str(i)
        
    def get_weight(self):
        m, n = self.weighted_sign.shape
        usage = np.zeros(n)
        
        for k in range(0, m):
            sel = [p for p in range(0, n) if not self.weighted_sign[k, p] == 0]
            usage[sel] += 1
        
        sel = [k for k in range(0, n) if not usage[k] == 0]
        self.weights = np.zeros(n)
        self.weights[sel] = m/usage[sel]
        self.weights[sel] = np.log(self.weights[sel])
        
        
    def weight_sign(self):
        for i in range (0, 180):
            sign = self.weighted_sign[i,:]
            
            #weight the signature
            L = sign.size
            sign = np.multiply(self.weights, sign)
            sign = sign/np.sum(sign)
            
            #find the cutoff point
            sorted_sign = np.sort(sign)
            part = np.cumsum(sorted_sign)
            part += np.array(range(L-1, -1, -1))*sorted_sign - sorted_sign/self.cutoff
            best = max([k for k in range(0, len(part)) if part[k] >= 0])
            thiscut = sorted_sign[best]
            
            if thiscut == 0:
                thiscut = max(thiscut, 1.0/L)
            
            #cut and quantize the signature
            sign_list = [k if k < thiscut else thiscut for k in sign.tolist()]
            sign = np.array(sign_list)
            sign = sign/np.sum(sign)
            sign = np.floor(255*sign/self.cutoff)
            rem = int(round(255/self.cutoff) - np.sum(sign))
            sel = [k for k in range(0, L) if not sign[k] == 0 and sign[k] < 255] +\
                    [p for p in range(0, L) if sign[p] == 0]
            sign[sel[0:rem]] += 1
            
            print sign
            
            self.weighted_sign[i,:] = sign
        
    def save_weighted_sign(self, file_dir, file_name):
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
            
        try:
            with open(os.path.join(file_dir, file_name), 'wb') as file_data:
                np.save(file_data, self.weighted_sign)
        except IOError as ioerror:
            print ioerror
        
if __name__ == '__main__':
    wt = Weight(0.01, 180, 10, 4)
     
    wt.get_weight()
    
    print wt.weights
    
    wt.weight_sign()
    
    wt.save_weighted_sign('./Signature/1000/', 'weighted_sign')
    
    
    
        
    
    
    
    
    
    