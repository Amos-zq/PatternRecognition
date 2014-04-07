'''
Created on Apr 7, 2014

@author: yulu
'''

class PatternRecognition:
    
    KEYPOINT_DIR = './Keypoint/'
    DESC_DIR = './Descriptor/'
    SIGN_DIR = './Signature/'
    TREE_DIR = './'
    TREE_FILE = 'tree.vlhkm'
    
    '''
    Build a signature database from a sets of images
    '''
    def Build_Database(self, 
                 img_dir, #directory to the images
                 num_in_set, #number of image in each set
                 total_set, #total number of sets
                 num_kpts, #number of kpt                            
                 K,#tree branch
                 depth, #tree depth
                 
                 cutoff #cutoff value for weighted sign
                ):
        
        self.total = total_img
        self.num = num_in_set
        self.num_kpts = num_kpts
        self.K = K
        self.depth = depth
        self.cutoff = cutoff
        
        self.L = (K**(depth+1)-1) / (K-1) - 1 #calculate the number of nodes in the tree
        
        #load random keypoint
        kpt = Keypoint()
        kpt.load_keypoint(keypoint_dir, keypoint_file)
        
        self.rkpt = kpt.kpt
        
        #load signature database
        sign = Signature()
        sign.load_sign(sign_dir, sign_file)
        
        self.sign_database = sign.sign_database