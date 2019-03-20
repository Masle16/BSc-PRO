import os
from glob import glob
import cv2
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt


def images_to_numpy(images_pot, images_cat, images_tab):
    x = []
    y = []
    
    width = 224
    height = 224
    
    for img_pot in images_pot:
        true_color_img = cv2.cvtColor(cv2.imread(img_pot),cv2.COLOR_BGR2RGB)
        x.append(cv2.resize(true_color_img, (width,height), interpolation=cv2.INTER_CUBIC))
        y.append([1,0,0])
    for img_cat in images_cat:
        true_color_img = cv2.cvtColor(cv2.imread(img_cat),cv2.COLOR_BGR2RGB)
        x.append(cv2.resize(true_color_img, (width,height), interpolation=cv2.INTER_CUBIC))
        y.append([0,1,0])
    for img_tab in images_tab:
        true_color_img = cv2.cvtColor(cv2.imread(img_tab),cv2.COLOR_BGR2RGB)
        x.append(cv2.resize(true_color_img, (width,height), interpolation=cv2.INTER_CUBIC))
        y.append([0,0,1])
    return np.asarray(x), np.asarray(y)