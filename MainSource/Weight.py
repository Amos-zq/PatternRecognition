'''
Created on Apr 5, 2014

@author: yulu
'''
import numpy as np
from Signature import Signature
import os.path

class Weight:
    def __init__(self, cutoff):
        self.cutoff = cutoff        
        
    def get_weight(self, sign_database):
        m, n = sign_database.shape
        usage = np.zeros(n)
        
        for k in range(0, m):
            sel = [p for p in range(0, n) if not sign_database[k, p] == 0]
            usage[sel] += 1
            
        #print usage.tolist()
        
        sel = [k for k in range(0, n) if not usage[k] == 0]
        self.weights = np.empty(n)
        self.weights[sel] = m/usage[sel]
        self.weights[sel] = np.log(self.weights[sel])
        
        print 'Calculate weights'
             
    def weight_train_database(self, sign_database):
        m, n = sign_database.shape
        self.weighted_sign = np.empty(shape=(m, n))
        
        for i in range (0, m):           
            self.weighted_sign[i,:] = self.weight_sign(sign_database[i, :])
                      
    def weight_sign(self, sign):       
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
        #print np.sum(sign)
        rem = int(round(255/self.cutoff) - np.sum(sign))
        sel = [k for k in range(0, L) if sign[k] > 0 and sign[k] < 255] +\
                [p for p in range(0, L) if sign[p] == 0]
        sign[sel[0:rem]] += 1
        
        #print "Weighting the Sign"
        return sign  
   
    def save_weights(self, file_dir, file_name):
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
            
        try:
            with open(os.path.join(file_dir, file_name), 'wb') as file_data:
                np.save(file_data, self.weights)
        except IOError as ioerror:
            print ioerror
       
    def save_weighted_sign(self, file_dir, file_name):
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
            
        try:
            with open(os.path.join(file_dir, file_name), 'wb') as file_data:
                np.save(file_data, self.weighted_sign)
        except IOError as ioerror:
            print ioerror            
    
    def load_weights(self,file_dir, file_name):
        try:
            with open(os.path.join(file_dir, file_name), 'rb') as file_data:
                self.weights = np.load(file_data)
        except IOError as ioerror:
            print ioerror 
        
    def load_weighted_sign(self, file_dir, file_name):
        try:
            with open(os.path.join(file_dir, file_name), 'rb') as file_data:
                self.weighted_sign = np.load(file_data)
        except IOError as ioerror:
            print ioerror     
        
        
if __name__ == '__main__':
         
    # load signature
    L = (10**(4+1)-1) / (10-1) - 1
    sign_database = np.empty(shape=(180, L))
    sign = Signature()
    sign.load_sign('./Signature/1000/', 'sign_1000')
    
    wt = Weight(0.01)    
    wt.get_weight(sign.sign_database)
    
    #print wt.weights  
    wt.weight_train_database(sign.sign_database)
    
    wt.save_weights('./Signature/1000/', 'weights')
    wt.save_weighted_sign('./Signature/1000/', 'weighted_sign')
    
    
    
        
    
    
    
    
    
    