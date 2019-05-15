###### IMPORTS ######
import os
import glob
from imutils import paths
import cv2
import numpy as np
import matplotlib.pyplot as plt
from knn_classifier import kNearestNeighbor

###### FUNCTIONS ######
def time_function(func, *args):
    """ 
    Call a function, func, with args and return the time, in [s],
    that it took to execute.
    """
    
    import time
    tic = time.time()
    func(*args)
    toc = time.time()
    
    return toc - tic

def extract_hist(image, bins=(8, 8, 8)):
    """
    Extracts a 3D color histogram from the hsv color space using
    the supplied number of bins per channel.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(images=[hsv],
                        channels=[0, 1, 2],
                        mask=None,
                        histSize=bins,
                        ranges=[0, 180, 0, 256, 0, 256])

    cv2.normalize(src=hist, dst=hist)
    return hist.flatten()

def image_to_feature_vector(image, size=(32, 32)):
    """
    resize the image to a fixed size, then flatten the image into
    a list of raw pixel intensities
    """

    return cv2.resize(image, size).flatten()

def convert_label_to_number(label):
    """ Converts label to number """

    result = None
    if label == 'bgd':
        result = 0
    elif label == 'potato':
        result = 1
    elif label == 'carrot':
        result = 2
    elif label == 'beef':
        result = 3
    elif label == 'sal':
        result = 4
    elif label == 'bun':
        result = 5
    elif label == 'arm':
        result = 6
    elif label == 'ketchup':
        result = 7
    return result

def cross_val(k_choices, num_folds, X_train_folds, y_train_folds, shape):
    """ Performs cross validation """
    
    # A dictionary holding the accuracies to find the best value of k
    k_to_accuracies = {}

    # Perform k-fold cross validation to find the best value of k
    for k in k_choices:
        for j in range(num_folds):
            all_but_one_ind = [i for i in range(num_folds) if i != j]
            X_all_but_one = np.concatenate(X_train_folds[all_but_one_ind])
            y_all_but_one = np.concatenate(y_train_folds[all_but_one_ind])

            knn = kNearestNeighbor()
            knn.train(X_all_but_one, y_all_but_one)
            y_pred_k_f = knn.predict(X_train_folds[j], k)

            acc = float(sum(y_pred_k_f == y_train_folds[j])) / shape

            if k not in k_to_accuracies:
                k_to_accuracies[k] = []
            k_to_accuracies[k].append(acc)

    # Plot the raw observations
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # Plot the trend line with error bars that corresponde to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

def print_info(X_train, y_train, X_test, y_test):
    """ Prints information """
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    print('Train matrix:', (X_train.nbytes / 1.0e6), 'MB')
    print('Test matrix:', (X_test.nbytes / 1.0e6), 'MB')
    print('Train data shape:', X_train.shape)
    print('Tain labels shape:', y_train.shape)
    print('Test data shape:', X_test.shape)
    print('Test labels shape:', y_test.shape)

    ###### CREATE AND TRAIN CLASSIFIER ######
    classifier = kNearestNeighbor()
    classifier.train(X_train, y_train)

    # Test implementation
    dists = classifier.compute_distances_no_loops(X_test)
    print('Distance shape:', dists.shape)
    plt.figure(figsize=(12, 9))
    plt.imshow(dists, interpolation='none')
    plt.show()

    # Time performance
    time = []
    for i in range(30):
        time.append(time_function(classifier.compute_distances_no_loops, X_test))
    
    print('Average time performance:', np.mean(time), 'seconds')
    
    return X_train, y_train, X_test, y_test

def plot_image_samples(images, y_test):
    """ Plot some images to display the dataset """

    classes = ['Table','Potato','Carrot','Sal','Beef','Bun','Arm','Ketchup']
    num_classes = len(classes)
    samples_per_class = 5
    plt.figure(figsize=(12, 9))
    for y, cls in enumerate(classes):
        idxs = [i for i, label in enumerate(y_test) if label == y]
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(images[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()
    
def multiband_threshold(img):
    """ Performs multiband thresholding """
        
    lower = (0, 65, 0)
    upper = (179, 255, 255)
    
    # Multiband thresholding
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(src=hsv, lowerb=lower, upperb=upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    img = cv2.bitwise_and(img, img, mask=mask)
    
    return img

def smooth(img, size=(25, 25)):
    """ Performs gaussian smoothing on image """
    
    return cv2.GaussianBlur(img, size, 0)