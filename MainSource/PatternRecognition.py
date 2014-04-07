'''
Created on Apr 7, 2014

@author: yulu
'''

import numpy as np
import vlfeat as vl
import os.path
from PIL import Image

from Keypoint import Keypoint
from Descriptor import Descriptor
from Signature import Signature
from Tree import Tree
from Weight import Weight

class PatternRecognition:
    
    KEYPOINT_DIR = './Keypoint/'
    KEYPOINT_FILE = 'kpt_'
    DESC_DIR = './Descriptor/'
    DESC_FILE = 'desc_'
    SIGN_DIR = './Signature/'
    SIGN_FILE = 'sign_'
    TREE_DIR = './Tree'
    TREE_FILE = '_tree.vlhkm'
    WEIGHT_FILE = 'weights_'
    WEIGHT_SIGN_FILE = './weighted_sign_'
    
    def dist(self, data_sign, query_sign):
        diff = data_sign-query_sign
        diff = np.fabs(diff)
        return np.sum(diff)
    
    def Build_Database(self, img_dir, num_of_sets, num_in_set):
        self.total = num_of_sets * num_in_set
        self.cla = num_of_sets
        self.size = num_in_set
        
        self.database = []
        set_index = 0
        for folder_name in os.listdir(img_dir):
            if set_index >= num_of_sets:
                break;
            index = 0;
            for file_name in os.listdir(os.path.join(img_dir, folder_name)): #later randomize this
                if index >= num_in_set:
                    break
                
                data_set = (index, set_index, os.path.join(img_dir, folder_name,file_name))
                
                index += 1
                
                self.database.append(data_set) 
                
            set_index += 1
    
    '''
    Build a signature database from a sets of images
    '''
    def Train_Database(self, 
                 img_dir, #directory to the images
                 img_width, img_height, #size of image
                 num_in_set, #number of image in each set
                 total_set, #total number of sets
                 num_kpts, #number of kpt                            
                 K,#tree branch
                 depth, #depth
                 nleaves #leaves in the tree
                ):
        
        self.total = num_in_set * total_set
        self.total_set = total_set
        self.num = num_in_set
        self.num_kpts = num_kpts
        self.K = K
        self.depth = depth
        
        '''
        generate keypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        '''
        kp = Keypoint()   
        kp.generate_keypoint(num_kpts, img_width, img_height, 1)   #sigma is set to 1 currently
        kp.save_keypoint(self.KEYPOINT_DIR, self.KEYPOINT_FILE+str(num_kpts)) #save to a directory
        
        print 'Random keypoint generated'
        
        
        '''
        generate desc~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        '''
        desc_database = np.empty(shape=(128, self.total*self.num_kpts))
        k = 0
        for folder_name in os.listdir(img_dir):
            count = 0;
            for file_name in os.listdir(os.path.join(img_dir, folder_name)):
                if count >= num_in_set:
                    break
                #load image(load image, convert to np array with float type)
                img = Image.open(os.path.join(img_dir, folder_name, file_name))
                img_data = np.asarray(img, dtype=float)
    
                #generate desc
                desc = Descriptor()
                desc.generate_desc(img_data, kp.kpt)
                desc.save_desc(os.path.join(self.DESC_DIR, str(self.num_kpts)), self.DESC_FILE + str(k))
                
                count += 1
                k += 1
                desc_database[:, k*num_kpts:k+num_kpts] = desc.desc #add to the database therefore later can be used to train the tree
                print '=>'+str(k) ,
        print 'Descriptor Generated'
        
        '''
        Build the tree~~~~~~~~~~~~~~~~~~~~~~~~~
        '''
       
        tree = Tree()
        tree.generate_tree(desc_database, self.K, self.nleaves, self.TREE_DIR, str(num_kpts)+self.TREE_FILE)
        
        print 'Tree built'
        
        '''
        Generate signature~~~~~~~~~~~~~~~~~~~~~~~~~
        '''
        tr = vl._vlfeat.VlHIKMTree(0, 0)
        tr.load(self.TREE_DIR + str(num_kpts)+self.TREE_FILE)
    
        sign = Signature()
        sign.generate_sign_database_dir(tr, desc_database, self.K, self.depth, self.total, self.num_kpts)
    
        sign.save_sign(self.SIGN_DIR, self.SIGN_FILE+str(self.num_kpts))
        
        print 'Signature Generated'
        
        del desc_database
        
    '''
    calculate the weights and weight the signature in database
    '''
    def Weight_Database(self,
                        num_kpts,
                        total_img,
                        K,
                        depth,
                        cutoff):
        #load signature
        sign = Signature()
        sign.load_sign(self.SIGN_DIR, self.SIGN_FILE+str(num_kpts))
        
        L = (K**(depth+1)-1) / (K-1) - 1
        wt = Weight(total_img, L, cutoff)
        wt.get_weight(sign.sign_database)        
        wt.weight_train_database(sign.sign_database)
    
        wt.save_weights(self.SIGN_DIR, self.WEIGHT_FILE+str(num_kpts))
        wt.save_weighted_sign(self.SIGN_DIR, self.WEIGHT_SIGN_FILE+str(num_kpts))
        
    def Classifier(self, num_kpts, img_dir, total_test_img, num_in_set, K, depth, cutoff, top=5):
        classify_result =[]
        #load tree and weight
        wt = Weight(cutoff)
        wt.load_weights('./Signature/1000/', 'weights')
        wt.load_weighted_sign('./Signature/1000/', 'weighted_sign')
    
        #Load tree
        tr = vl._vlfeat.VlHIKMTree(0, 0)
        tr.load(self.TREE_DIR+str(num_kpts)+self.TREE_FILE)
        

        for (k, image_name) in self.database:
            #randomly get image from the img_dir
            img = Image.open(os.path.join(folder_name, file_name))
            img_data = np.asarray(img, dtype=float)
            
            #generate desc, sign and weighted sign
            kp = Keypoint()
            kp.load_keypoint(self.KEYPOINT_DIR, self.KEYPOINT_FILE+str(num_kpts))
            desc = Descriptor()
            desc.generate_desc(img_data, kp.kpt)
            sign = Signature()
            s = sign.generate_sign(tr, desc.desc, K, depth)
            weighted_sign = wt.weight_sign(s)
            
            #vote
            d=np.empty(self.total_img)
            for i in range(0, self.total_img):   
                d[i] = self.dist(wt.weighted_sign[i,:], weighted_sign)
        
            perm = np.argsort(d)
            vote_for = np.floor((perm[0:top])/num_in_set)+1                      
            votes = vl.vl_binsum(np.zeros(self.total_set), np.ones(top), vote_for)
            
            #print votes
            best = np.argmax(votes)
            
            classify_result[k] = best
            
if __name__ =='__main__':
    PR = PatternRecognition()
    PR.Build_Database('./Image/', 15, 8)
    
    print PR.database
        