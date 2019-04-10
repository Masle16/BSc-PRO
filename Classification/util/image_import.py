import os
from os import listdir
from os.path import isfile, join
from glob import glob
import cv2
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt

def get_sub_dir(path, ignore=[]):
    sub_directories = []
    list_sub_dir = listdir(path)
    # Removes folder which is in ignore
    for i in ignore:
        list_sub_dir.remove(i)
        
    for sub_dir in list_sub_dir:
        sub_directories.append(glob(os.path.join(path + "/" + sub_dir, "*.jpg")))
        
    return sub_directories

def calulate_mean(path_training_img, ignore=[]):
    width = 224
    height = 224
    mean_img = np.zeros((width, height, 3), dtype="float32")
    numbers_img = 0
    
    sub_dirs = get_sub_dir(path_training_img, ignore)

    for sub in sub_dirs:
        for img in sub:
            temp_img = cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB)
            mean_img += cv2.resize(temp_img, (width,height), interpolation=cv2.INTER_CUBIC)
            numbers_img += 1
            
    return (mean_img/numbers_img)

##### TEST ####### Calculate mean pr. channel over all pictures

def calculate_mean_keras(x):
    mean = np.mean(x, axis=(0, 1, 2))
    broadcast_shape = [1, 1, 1]
    broadcast_shape[3 - 1] = x.shape[3]
    mean = np.reshape(mean, broadcast_shape)
    return mean

#################

def images_to_numpy(images_pot, images_cat, images_tab):
    x = []
    y = []
    
    width = 224
    height = 224
    
    for img_pot in images_pot:
        true_color_img = cv2.cvtColor(cv2.imread(img_pot),cv2.COLOR_BGR2RGB)
        x.append(cv2.resize(true_color_img, (width,height), interpolation=cv2.INTER_CUBIC))
        y.append(0)
    for img_cat in images_cat:
        true_color_img = cv2.cvtColor(cv2.imread(img_cat),cv2.COLOR_BGR2RGB)
        x.append(cv2.resize(true_color_img, (width,height), interpolation=cv2.INTER_CUBIC))
        y.append(1)
    for img_tab in images_tab:
        true_color_img = cv2.cvtColor(cv2.imread(img_tab),cv2.COLOR_BGR2RGB)
        x.append(cv2.resize(true_color_img, (width,height), interpolation=cv2.INTER_CUBIC))
        y.append(2)
    return np.asarray(x), np.asarray(y)

def images_to_numpy_full_class(image_path, ignore=[]):
    x = []
    y = []
    
    width = 224
    height = 224
    sub_dirs = get_sub_dir(image_path, ignore)
    for label, classes in enumerate(sub_dirs):
        for img in classes:
            true_color_img = cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB)
            x.append(cv2.resize(true_color_img, (width,height), interpolation=cv2.INTER_CUBIC))
            y.append(label)
    x = np.asarray(x)
    x = np.float32(x)
    return x, y

def images_import(images_pot, images_cat, images_tab):
    x = []
    for img_pot in images_pot:
        x.append(plt.imread(img_pot))
    for img_cat in images_cat:
        x.append(plt.imread(img_cat))
    for img_tab in images_tab:
        x.append(plt.imread(img_tab))
    return np.asarray(x)

def make_images_mean(X_train, path_from, path_to):
    mean_image = np.mean(X_train, axis=0)
    
    # Imports picture path
    test_pot = glob(os.path.join((path_from + "/test/potato"),"*.jpg"))
    test_cat = glob(os.path.join((path_from + "/test/catfood"),"*.jpg"))
    test_tab = glob(os.path.join((path_from + "/test/table"), "*.jpg"))
   
    train_pot = glob(os.path.join((path_from + "/train/potato"),"*.jpg"))
    train_cat = glob(os.path.join((path_from + "/train/catfood"),"*.jpg"))
    train_tab = glob(os.path.join((path_from + "/train/table"), "*.jpg"))
    
    valid_pot = glob(os.path.join((path_from + "/valid/potato"),"*.jpg"))
    valid_cat = glob(os.path.join((path_from + "/valid/catfood"),"*.jpg"))
    valid_tab = glob(os.path.join((path_from + "/valid/table"), "*.jpg"))

    for index, img in enumerate(test_pot):
        img = plt.imread(img) - mean_image
        scipy.misc.imsave(path_to+"/test/potato/potato"+str(index)+".jpg", img)
    for index, img in enumerate(test_cat):
        img = plt.imread(img) - mean_image
        scipy.misc.imsave(path_to+"/test/catfood/catfood"+str(index)+".jpg", img)
    for index, img in enumerate(test_tab):
        img = plt.imread(img) - mean_image
        scipy.misc.imsave(path_to+"/test/table/table"+str(index)+".jpg", img)
        
    for index, img in enumerate(train_pot):
        img = plt.imread(img) - mean_image
        scipy.misc.imsave(path_to+"/train/potato/potato"+str(index)+".jpg", img)
    for index, img in enumerate(train_cat):
        img = plt.imread(img) - mean_image
        scipy.misc.imsave(path_to+"/train/catfood/catfood"+str(index)+".jpg", img)
    for index, img in enumerate(train_tab):
        img = plt.imread(img) - mean_image
        scipy.misc.imsave(path_to+"/train/table/table"+str(index)+".jpg", img)
        
    for index, img in enumerate(valid_pot):
        img = plt.imread(img) - mean_image
        scipy.misc.imsave(path_to+"/valid/potato/potato"+str(index)+".jpg", img)
    for index, img in enumerate(valid_cat):
        img = plt.imread(img) - mean_image
        scipy.misc.imsave(path_to+"/valid/catfood/catfood"+str(index)+".jpg", img)
    for index, img in enumerate(valid_tab):
        img = plt.imread(img) - mean_image
        scipy.misc.imsave(path_to+"/valid/table/table"+str(index)+".jpg", img)
    return mean_image, plt.imread(train_pot[0])
                 
# Downscales and convert image from BGR to RGB into a array
def images_to_arr(images):
    x = []
    width = 224
    height = 224
    
    for img in images:
        true_color_img = cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB)
        x.append(cv2.resize(true_color_img, (width,height), interpolation=cv2.INTER_CUBIC))
    return x # Makes to numpy array

def get_data(x_kar, x_kat, x_bord, y_kar=0, y_kat=1, y_bord=2, 
                           num_training=205, num_validation=100, num_test=100):
        X_total = []
        y_total = []
        
        for kar in x_kar:
            X_total.append(kar)
            y_total.append(y_kar)
        for kat in x_kat:
            X_total.append(kat)
            y_total.append(y_kat)
        for bord in x_bord:
            X_total.append(bord)
            y_total.append(y_bord)
        X_total = np.asarray(X_total) # Make a numpy array
        y_total = np.asarray(y_total)
        X_total = X_total.astype(float)
        X_total = np.divide(X_total,255)
        # Generate X_total.shape[0] num from X_total.shape[0]
        mask = np.random.choice(X_total.shape[0], X_total.shape[0], replace=True) 
        mask_train = mask[0:num_training]
        mask_val   = mask[num_training:num_training+num_validation]
        mask_test  = mask[num_training+num_validation:num_training+num_validation+num_test]
        
        # Creating training, validation and test based on maskes
        X_train = X_total[mask_train]
        y_train = y_total[mask_train]
        X_val = X_total[mask_val]
        y_val = y_total[mask_val]
        X_test = X_total[mask_test]
        y_test = y_total[mask_test]
        
        
        # Normalize the data: subtract the mean image
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
        
        # Make it into one vector
        X_train = X_train.reshape(num_training, -1)
        X_val = X_val.reshape(num_validation, -1)
        X_test = X_test.reshape(num_test, -1)
        
        return X_train, y_train, X_val, y_val, X_test, y_test

def get_data_aug(x_kar, x_kat, x_bord, x_kar_aug, x_kat_aug, x_bord_aug, y_kar=0, y_kat=1, y_bord=2, 
                           num_training=205, num_validation=100, num_test=100):
        X_total = []
        y_total = []
        X_augmented = []
        y_augmented = []
        # Running through source images
        for kar in x_kar:
            X_total.append(kar)
            y_total.append(y_kar)
        for kat in x_kat:
            X_total.append(kat)
            y_total.append(y_kat)
        for bord in x_bord:
            X_total.append(bord)
            y_total.append(y_bord)
            
        # Running through augmented images
        for kar in x_kar_aug:
            X_augmented.append(kar)
            y_augmented.append(y_kar)
        for kat in x_kat_aug:
            X_augmented.append(kat)
            y_augmented.append(y_kat)
        for bord in x_bord_aug:
            X_augmented.append(bord)
            y_augmented.append(y_bord)
        
        # Make a numpy array
        X_total = np.asarray(X_total) 
        y_total = np.asarray(y_total)
        
        X_total = X_total.astype(float)
        X_total = np.divide(X_total,255)
        
        # Generate maskes to randomize data selection
        mask_source = np.random.choice(X_total.shape[0], X_total.shape[0], replace=True) 
        mask_train = mask_source[0:num_training]
        mask_val   = mask_source[num_training:num_training+num_validation]
        mask_test  = mask_source[num_training+num_validation:num_training+num_validation+num_test]
        
        # Combining source and augmented images for training data
        X_source_train = X_total[mask_train]
        y_source_train = y_total[mask_train]
        
        for data in X_source_train:
            X_augmented.append(data)
        for data in y_source_train:
            y_augmented.append(data)
        
        X_augmented = np.asarray(X_augmented)
        y_augmented = np.asarray(y_augmented)
        
        X_augmented = X_augmented.astype(float)
        X_augmented = np.divide(X_augmented,255)
        
        mask_augmented = np.random.choice(X_augmented.shape[0], X_augmented.shape[0])
        # Creating training, validation and test based on maskes
        X_train = X_augmented[mask_augmented]
        y_train = y_augmented[mask_augmented]
        X_val = X_total[mask_val]
        y_val = y_total[mask_val]
        X_test = X_total[mask_test]
        y_test = y_total[mask_test]

        # Normalize the data: subtract the mean image
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
        
        # Make it into one vector
        X_train = X_train.reshape(X_augmented.shape[0], -1)
        X_val = X_val.reshape(num_validation, -1)
        X_test = X_test.reshape(num_test, -1)
        
        return X_train, y_train, X_val, y_val, X_test, y_test

def import_images(path_potato, path_catfood, path_table, augmented=False, path_aug_potato=None, path_aug_catfood=None,
                  path_aug_table=None):
    
        x_pot = images_to_arr(path_potato) 
        x_cat = images_to_arr(path_catfood)
        x_tab = images_to_arr(path_table)
        if augmented:
            x_aug_pot = images_to_arr(path_aug_potato[0:500])
            x_aug_cat = images_to_arr(path_aug_catfood[0:500])
            x_aug_tab = images_to_arr(path_aug_table[0:500])
            X_train, y_train, X_val, y_val, X_test, y_test = get_data_aug(x_pot, x_cat, x_tab, x_aug_pot, x_aug_cat, x_aug_tab)
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = get_data(x_pot, x_cat, x_tab)
        return X_train, y_train, X_val, y_val, X_test, y_test
                                                                       