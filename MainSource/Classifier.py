'''
Created on Apr 6, 2014

@author: yulu
'''
import numpy as np
import vlfeat as vl

from MainSource.Keypoint import Keypoint
from MainSource.Descriptor import Descriptor
from MainSource.Signature import Signature
from MainSource.Weight import Weight
import os.path
import pickle
from PIL import Image, ImageOps

DATABASE_DIR='/home/yulu/Research/Web/aptana_workspace/PatternRecognition/MainSource/Database/'

KEYPOINT_DIR = '/home/yulu/Research/Web/aptana_workspace/PatternRecognition/MainSource/Keypoint/'
KEYPOINT_FILE = 'kpt_'

DESC_DIR = './Descriptor/'
DESC_FILE = 'desc_'

TREE_DIR = '/home/yulu/Research/Web/aptana_workspace/PatternRecognition/MainSource/Tree'
TREE_FILE = '_tree.vlhkm'

SIGN_DIR = '/home/yulu/Research/Web/aptana_workspace/PatternRecognition/MainSource/Signature/'
SIGN_FILE = 'sign_'

WEIGHT_FILE = 'weights_'
WEIGHT_SIGN_FILE = 'weighted_sign_'

SIGMA = 1

def dist(data_sign, query_sign):
    diff = data_sign-query_sign
    diff = np.fabs(diff)
    return np.sum(diff)

def StandalizeImage(img, h):
    [width, height] = img.size
    ratio = float(height)/h
    w = int(width/ratio)
    img = ImageOps.fit(img, [w, h] , Image.ANTIALIAS)
    
    return img

def Classifier(image_path, database_version):
    print os.path.abspath(DATABASE_DIR)
    
    #load database info   
    try:
        with open(os.path.join(DATABASE_DIR, 'database_'+ str(database_version)), 'rb') as rfile:
            database = pickle.load(rfile)
    except IOError as ioerr:
        print ioerr
        
    num_in_set = database[0]
    num_of_sets = database[1]
    data = database[2]
    total = num_in_set * num_of_sets
    cutoff = 0.01
    num_of_kpts = 2000
    top = 1
    
    #Load weight
    wt = Weight(cutoff)
    sign_dir = os.path.join(SIGN_DIR, 'db_version_' + str(database_version))
    wt.load_weights(sign_dir, WEIGHT_FILE+str(num_of_kpts)+'_'+str(cutoff))
    wt.load_weighted_sign(sign_dir, WEIGHT_SIGN_FILE+str(num_of_kpts)+'_'+str(cutoff))
    
    #Load tree
    tree_dir = os.path.join(TREE_DIR, 'db_version_' + str(database_version))
    tr = vl._vlfeat.VlHIKMTree(0, 0)
    tr.load(os.path.join(tree_dir, str(num_of_kpts) + TREE_FILE))
    
    '''classify the input image'''
    #randomly get image from the img_dir
    img = Image.open(image_path).convert('L')
    img = StandalizeImage(img, 480)
    img_data = np.asarray(img, dtype=float)
    
    #generate desc, sign and weighted sign
    kp = Keypoint()
    #kp.load_keypoint(self.KEYPOINT_DIR, self.KEYPOINT_FILE+str(num_of_kpts))
    kp.generate_keypoint(num_of_kpts, img.size[0], img.size[1], SIGMA)
    desc = Descriptor()
    desc.generate_desc(img_data, kp.kpt)
    
    sign = Signature()
    s = sign.generate_sign(tr,desc.desc, 10, 4)
    weighted_sign = wt.weight_sign(s)
    
    #vote
    d=np.empty(total)
    for i in range(0, total):   
        d[i] = dist(wt.weighted_sign[i,:], weighted_sign)
    
    perm = np.argsort(d)
    vote_for = np.floor((perm[0:top])/num_in_set)+1                      
    votes = vl.vl_binsum(np.zeros(num_of_sets), np.ones(top), vote_for)
    
    #print votes
    best = np.argmax(votes)
    
    #get class name from folder name
    class_name = data[best*num_in_set][2]
    class_name = class_name.split("/")[2]
    
    return best, class_name
    
    

if __name__ == '__main__':
    test_image_path = './Image/u_town/stand-image_03.jpg'
    idx, class_folder = Classifier(test_image_path, 1)
     
    print idx, class_folder
        
        
