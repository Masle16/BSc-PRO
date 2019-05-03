# IMPORTS
import cv2
import os
from os import listdir
from glob import glob

# FUNCTIONS
def get_sub_dir(path, ignore=[]):
    sub_directories = []
    list_sub_dir = listdir(path)
    # Removes folder which is in ignore
    for i in ignore:
        list_sub_dir.remove(i)
        
    for sub_dir in list_sub_dir:
        sub_directories.append(glob(os.path.join(path + "/" + sub_dir, "*.jpg")))
        
    return sub_directories

def subtrack_and_save(load_path, save_path, filter_path):
    sub_dir = get_sub_dir(load_path)
    img_subtrack = None
    img_filter = cv2.imread(filter_path, cv2.IMREAD_COLOR)
    numofimg = 0

    for cat in sub_dir:
        for img in cat:
            img_subtrack = cv2.imread(img, cv2.IMREAD_COLOR) # Load in one img
            img_subtrack = cv2.bitwise_and(img_subtrack, img_filter) # bit_wice and
            path = img.replace(load_path,save_path)
            numofimg += 1
            print(numofimg)
            cv2.imwrite(path,img_subtrack)

# MAIN
save_path_trian = 'dataset3/img_filter/train'
save_path_test = 'dataset3/img_filter/test'

load_path_trian = 'dataset3/res_still/train'
load_path_test = 'dataset3/res_still/test'

filter_path = 'preprocessing/bgd_mask.JPG'

subtrack_and_save(load_path_test, save_path_test, filter_path)
subtrack_and_save(load_path_trian, save_path_trian, filter_path)