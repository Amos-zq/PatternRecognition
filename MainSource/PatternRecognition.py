'''
Created on Apr 7, 2014

@author: yulu
'''

import numpy as np
import vlfeat as vl
import os.path
from PIL import Image
import pickle

from Keypoint import Keypoint
from Descriptor import Descriptor
from Signature import Signature
from Tree import Tree
from Weight import Weight

import random


class PatternRecognition:
    
    DATABASE_DIR='./Database/'
    
    KEYPOINT_DIR = './Keypoint/'
    KEYPOINT_FILE = 'kpt_'
    
    DESC_DIR = './Descriptor/'
    DESC_FILE = 'desc_'
    
    TREE_DIR = './Tree'
    TREE_FILE = '_tree.vlhkm'
    
    SIGN_DIR = './Signature/'
    SIGN_FILE = 'sign_'

    WEIGHT_FILE = 'weights_'
    WEIGHT_SIGN_FILE = './weighted_sign_'
    
    WIDTH = 640
    HEIGHT = 480
    SIGMA = 1

    
    def dist(self, data_sign, query_sign):
        diff = data_sign-query_sign
        diff = np.fabs(diff)
        return np.sum(diff)
    
    def RandomSelection(self, input_list, sel):
        if sel > len(input_list):
            sel = len(input_list)
            
        r_list = []
        r_set = set()
        while len(r_list) < sel:
            rand_n = random.randint(0, len(input_list)-1)
            if rand_n not in r_set:
                r_set.add(rand_n)
                r_list.append(rand_n)
        return r_list
    
    '''
    specify the class index and path to the image giving a img directory
    a version num is given, the trained database should access this table by giving the version and number
    and also contains this num in its
    '''
    
    def Build_Database(self, 
                       img_dir, 
                       num_in_set, #number of image in each train set
                       num_in_test_set, #number of image in each test set
                       num_of_sets, #total number of sets
                       version #version num to store the database
                       ):
        
        if not os.path.isfile(os.path.join(PR.DATABASE_DIR, 'database_'+str(version))):
            print 'generate database'
            database = [] #train database
            test_database = [] #test database
            set_index = 0
            for folder_name in os.listdir(img_dir):
                
                if set_index >= num_of_sets:
                    break;
                
                file_list = os.listdir(os.path.join(img_dir, folder_name))
                
                sel = self.RandomSelection(range(0, len(file_list)), num_in_set) #random selection from the files in the dir
                sel_test = self.RandomSelection(range(0, len(file_list)), num_in_test_set)
                
                for i in range(0, num_in_set):                  
                    file_name = file_list[sel[i]]                  
                    data_set = [i, set_index, os.path.join(img_dir, folder_name, file_name)]
                    
                    database.append(data_set) 
                    
                
                for j in range(0, num_in_test_set):
                    file_name = file_list[sel_test[j]]
                    test_data_set = [j, set_index, os.path.join(img_dir, folder_name, file_name)]
                    
                    test_database.append(test_data_set)
                    
                    
                set_index += 1
                
            #save database
            db_to_save = [num_in_set, num_of_sets, database, test_database]
            if not os.path.isdir(self.DATABASE_DIR):
                os.mkdir(self.DATABASE_DIR)
                
            try:
                with open(os.path.join(self.DATABASE_DIR, 'database_'+str(version)), 'wb') as wfile:
                    pickle.dump(db_to_save, wfile)
            except IOError as ioerr:
                print ioerr  
                     
        else:
            print 'database version exist, cannot overwrite for data protection, try another version number'
        
        
            
    
    '''
    Build a signature database from a batabase table
    '''
    def Train_Database_Sign(self, 
                 database,
                 version,
                 num_of_kpts,
                 force_update = False,                          
                 K = 10,#tree branch
                 depth = 4, #depth
                 nleaves = 10000 #leaves in the tree
                 ):
                    
        #load information from database
        num_in_set = database[0]
        num_of_sets = database[1]
        data_dir = database[2]   
        total = num_in_set * num_of_sets

        updated = force_update # will be turned true if one of the steps has been processed, so that the following step will be focused to processs!
        
        '''
        generate keypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        if file exists, load the kpt, otherwise generate it and save
        '''
        kp = Keypoint()   
        kp_file_name = os.path.join(self.KEYPOINT_DIR, self.KEYPOINT_FILE+str(num_of_kpts))
        if not force_update and os.path.isfile(kp_file_name):
            kp.load_keypoint(self.KEYPOINT_DIR, self.KEYPOINT_FILE+str(num_of_kpts))

            print 'Random keypoint loaded'
        else:
            kp.generate_keypoint(num_of_kpts, self.WIDTH, self.HEIGHT, self.SIGMA)   #sigma is set to 1 currently
            kp.save_keypoint(self.KEYPOINT_DIR, self.KEYPOINT_FILE+str(num_of_kpts)) #save to a directory
            updated = True
            print 'Random keypoint generated'
        
        
        '''
        generate desc~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ dir should be saved
        '''
        
        desc_dir = os.path.join(self.DESC_DIR, 'db_version_'+str(version), str(num_of_kpts))
        if updated or (not os.path.isdir(desc_dir) or os.listdir(desc_dir) == []):
            updated = True
            
            if not os.path.isdir(os.path.join(self.DESC_DIR, 'db_version_'+str(version))):
                os.mkdir(os.path.join(self.DESC_DIR, 'db_version_'+str(version)))
            if not os.path.isdir(desc_dir):
                os.mkdir(desc_dir)
            
            for d in data_dir:  
                img_idx = d[0]         
                class_idx = d[1]
                img_dir = d[2]
                
                '''
                k is the index of desc included in the name, should be ordered by the class 
                '''
                k = img_idx + class_idx * num_in_set
                
                #load image(load image, convert to np array with float type)
                img = Image.open(img_dir)
                img_data = np.asarray(img, dtype=float)
        
                #generate desc
                desc = Descriptor()
                desc.generate_desc(img_data, kp.kpt)
                desc.save_desc(desc_dir, self.DESC_FILE + str(k))
    
                print desc.desc
                
                #load to a large matrix
                #desc_database[:, k*num_of_kpts:k+num_of_kpts] = desc.desc #add to the database therefore later can be used to train the tree
                print '=>'+str(k) ,
                
            print 'Descriptor Generated'
            
                        
        #load desc 
        desc_database = np.empty(shape=(128, total*num_of_kpts))
        for k in range(0, total):
            desc = Descriptor()
            desc.load_desc(desc_dir, self.DESC_FILE + str(k))
            desc_database[:, k*num_of_kpts:(k+1)*num_of_kpts] = desc.desc
            
        print 'Descriptor Loaded'
        
        '''
        Build the tree~~~~~~~~~~~~~~~~~~~~~~~~~
        '''
        tree_dir = os.path.join(self.TREE_DIR, 'db_version_' + str(version))
        if updated or (not os.path.isfile(os.path.join(tree_dir, str(num_of_kpts) + self.TREE_FILE))): 
            updated = True
            if not os.path.isdir(tree_dir):
                os.mkdir(tree_dir)
                      
            tree = Tree()
            tree.generate_tree(desc_database, K, nleaves, tree_dir, str(num_of_kpts) + self.TREE_FILE)
        
            print 'Tree built'
   

        '''
        Generate signature~~~~~~~~~~~~~~~~~~~~~~~~~
        '''      
            
        sign_dir = os.path.join(self.SIGN_DIR, 'db_version_' + str(version))
        
        if updated or (not os.path.isfile(os.path.join(sign_dir, self.SIGN_FILE+str(num_of_kpts)))):
            updated = True
            
            tr = vl._vlfeat.VlHIKMTree(0, 0)
            tr.load(os.path.join(tree_dir, str(num_of_kpts) + self.TREE_FILE))
            
            print 'Tree Loaded'
            
            sign = Signature()
            sign.generate_sign_database_dir(tr, desc_database, K, depth, total, num_of_kpts)
    
            if not os.path.isdir(sign_dir):
                os.mkdir(sign_dir)
                
            sign.save_sign(sign_dir, self.SIGN_FILE+str(num_of_kpts))
            
            print 'Signature Generated'
        
        else:
            print 'Signature Already Generated'
        
        del desc_database
        
        return updated;
        
        
    '''
    calculate the weights and weight the signature in database
    '''
    def Build_Weight_Database(  self,
                                database,
                                version,
                                num_of_kpts,
                                cutoff,
                                force_update = False,
                                K = 10,
                                depth = 4,
                                nleaves = 10000):
                

        
        updated = self.Train_Database_Sign(database,version, num_of_kpts, force_update, K, depth, nleaves)
        
        #load signature
        sign_dir = os.path.join(self.SIGN_DIR, 'db_version_' + str(version))
        sign = Signature()
        sign.load_sign(sign_dir, self.SIGN_FILE+str(num_of_kpts))
        
        print 'Siganture Loaded'
        
        wt_file = os.path.join(sign_dir, self.WEIGHT_FILE+str(num_of_kpts)+'_'+str(cutoff))
        wts_file = os.path.join(sign_dir, self.WEIGHT_SIGN_FILE+str(num_of_kpts)+'_'+str(cutoff))
        
        if updated or (not os.path.isfile(wt_file) and not os.path.isfile(wts_file)):
            wt = Weight(cutoff)
            wt.get_weight(sign.sign_database)        
            wt.weight_train_database(sign.sign_database)
        
            wt.save_weights(sign_dir, self.WEIGHT_FILE+str(num_of_kpts)+'_'+str(cutoff))
            wt.save_weighted_sign(sign_dir, self.WEIGHT_SIGN_FILE+str(num_of_kpts)+'_'+str(cutoff))
            
            print ' '
            print 'Wegihted Sign Generated'
        
        else:
            print 'Weighted Sign Has Already Been Generated'
            
            
        
        
    def Classifier(self, 
                   database,
                   version,
                   num_of_kpts,
                   cutoff,
                   top = 5,
                   K = 10,
                   depth=4):
        
        num_in_set = database[0]
        num_of_sets = database[1]
        test_database = database[3]  
        
        total = num_in_set * num_of_sets
        num_in_test_set = len(test_database) / num_of_sets 
        
        classify_result =np.zeros(num_of_sets)
        #load tree and weight
        wt = Weight(cutoff)
        sign_dir = os.path.join(self.SIGN_DIR, 'db_version_' + str(version))
        wt.load_weights(sign_dir, self.WEIGHT_FILE+str(num_of_kpts)+'_'+str(cutoff))
        wt.load_weighted_sign(sign_dir, self.WEIGHT_SIGN_FILE+str(num_of_kpts)+'_'+str(cutoff))
    
        #Load tree
        tree_dir = os.path.join(self.TREE_DIR, 'db_version_' + str(version))
        tr = vl._vlfeat.VlHIKMTree(0, 0)
        tr.load(os.path.join(tree_dir, str(num_of_kpts) + self.TREE_FILE))    

        for k in test_database:
            #randomly get image from the img_dir
            img = Image.open(k[2])
            img_data = np.asarray(img, dtype=float)
            
            #generate desc, sign and weighted sign
            kp = Keypoint()
            kp.load_keypoint(self.KEYPOINT_DIR, self.KEYPOINT_FILE+str(num_of_kpts))
            desc = Descriptor()
            desc.generate_desc(img_data, kp.kpt)
            #very important !! convert desc to float type
            desc_f = np.array(desc.desc, dtype=float)
            
            sign = Signature()
            s = sign.generate_sign(tr,desc_f, K, depth)
            weighted_sign = wt.weight_sign(s)
            
            #vote
            d=np.empty(total)
            for i in range(0, total):   
                d[i] = self.dist(wt.weighted_sign[i,:], weighted_sign)
            
            perm = np.argsort(d)
            vote_for = np.floor((perm[0:top])/num_in_set)+1                      
            votes = vl.vl_binsum(np.zeros(num_of_sets), np.ones(top), vote_for)
            
            #print votes
            best = np.argmax(votes)
            
            if best == k[1]:
                classify_result[k[1]] += 1
                
            print '=>'+str(k[0])
        
        classify_result = classify_result / num_in_test_set
        
        print classify_result
        
            
if __name__ =='__main__':
    num_of_sets = 15
    num_of_image_in_test = 8
    num_of_image = 10
    image_folder = './Image/'
    version = 3
    
    PR = PatternRecognition()
    
    #generate the training database
    PR.Build_Database(image_folder, num_of_image, num_of_image_in_test, num_of_sets, version)
    
    #load the training database   
    try:
        with open(os.path.join(PR.DATABASE_DIR, 'database_'+ str(version)), 'rb') as rfile:
            database = pickle.load(rfile)
    except IOError as ioerr:
        print ioerr


    num_of_kpts = 2000
    cutoff = 0.01
    #if buid with another database version, indicate a force update!!
    #PR.Build_Weight_Database(database,version,num_of_kpts, cutoff)
    
    #PR.Classifier(database, version, num_of_kpts, cutoff, 1)
    
    
    
        