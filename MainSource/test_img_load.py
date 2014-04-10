'''
Created on Apr 10, 2014

@author: yulu
'''

import os.path

if __name__ == '__main__':
    l = os.listdir("./Image_large/")
    total_size = []
    
    for file_name in l:
        img_list = os.listdir(os.path.join("./Image_large",file_name))
        size = len(img_list)
        total_size.append(size)
        
    
    print total_size
    print min(total_size)