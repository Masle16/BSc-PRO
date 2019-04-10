#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

""" Module for k Nearest Neighbour """

###### IMPORTS ######
import os
from imutils import paths
import cv2
import numpy as np
import matplotlib.pyplot as plt

###### GLOBAL VARIABLES ######
NUM_TEST = 466
NUM_TRAINING = 1858

###### CLASSES ######
class kNearestNeighbor(object):
    """ A kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """ Train the classifier """

        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """ Predict labels for test data using this classifier """

        dists = self.compute_distances_no_loops(X)

        return self.predict_labels(dists, k=k)

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X
        and each training point in self.X_train using no explicit loops
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        squared_dist = np.sum(np.square(X)[:, np.newaxis, :], axis=2) - 2 * X.dot(self.X_train.T) + np.sum(np.square(self.X_train), axis=1)
        dists = np.sqrt(squared_dist)

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        """

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            # A list of length k storing the labels of the
            # k nearest neighbors to i'th test point

            closet_y = []

            min_index = np.argsort(dists[i])
            for j in range(k):
                closet_y.append(self.y_train[min_index[j]])

            y_pred[i] = np.bincount(closet_y).argmax()

        return y_pred

###### FUNCTIONS ######
def time_function(func, *args):
    """ Call a function f with args and return the time [s] that is took to execute """

    import time
    tic = time.time()
    func(*args)
    toc = time.time()

    return toc - tic

def image_to_feature_vector(image, size=(32, 32)):
    """ Rezise the image to a fixed size, then flatten the image into
    a list of raw pixel intensities. """

    return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Extract a 3D color histogram from the hsv color space using
    the supplied number of bins per channel
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(images=[hsv],
                        channels=[0, 1, 2],
                        mask=None,
                        histSize=bins,
                        ranges=[0, 180, 0, 256, 0, 256])

    # Normalize the histogram
    cv2.normalize(src=hist, dst=hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()

def convert_label_to_number(label):
    """ Converts label to number """

    result = None

    if label == 'background':
        result = 0
    elif label == 'potato':
        result = 1
    elif label == 'carrots':
        result = 2
    elif label == 'catfood_beef':
        result = 3
    elif label == 'catfood_salmon':
        result = 4
    elif label == 'bun':
        result = 5
    elif label == 'arm':
        result = 6
    elif label == 'ketchup':
        result = 7

    return result

def import_data(path):
    """ Import data """

    # Grab images from folder
    image_paths = list(paths.list_images(path))

    X = []
    y = []

    print('\n')
    for i, image_path in enumerate(image_paths):
        # Load the image and extract the class label
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = image_path.split(os.path.sep)[-1].split('.')[0]
        label = convert_label_to_number(label)

        # Extract histogram features from image
        hist = extract_color_histogram(image)

        # Update the features and labels matricies respectively
        X.append(hist)
        y.append(label)

        # Show an update every 100 images
        if i > 0 and i % 100 == 0:
            print('[INFO] processed {}/{}'.format(i, len(image_paths)))

    # Show some information on the memory consumed by the raw images
    # matrix and features matrix
    X = np.array(X)
    y = np.array(y)
    print('\n[INFO] features matrix: {:.2f}MB'.format(X.nbytes / (1024 * 1000.0)))

    # As a sanity check, we print out the size of the training and test data.
    print('\nData shape: ', X.shape)
    print('Labels shape: ', y.shape)

    return (X, y)

###### MAIN FUNCTION ######
def main():
    """ Main function """

    # Import train data and labels
    path = 'knn/train/'
    (X_train, y_train) = import_data(path)

    # Import test data and labels
    path = 'knn/test/'
    (X_test, y_test) = import_data(path)

    classifier = kNearestNeighbor()
    classifier.train(X_train, y_train)

    # Test your implementation
    dists = classifier.compute_distances_no_loops(X_test)
    print('\nDistance shape:', dists.shape)

    # Show the distance matrix (each row is a single test example and 
    # its distances to training examples)
    plt.imshow(dists, interpolation='none')
    plt.show()

    # Show time performance
    time = time_function(classifier.compute_distances_no_loops, X_test)
    print('\nTook %f seconds' % time)

    # Cross validation
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = []
    y_train_folds = []

    # Split up the training data into folds
    X_train_folds = np.array(np.array_split(X_train, num_folds))
    y_train_folds = np.array(np.array_split(y_train, num_folds))

    # A dictionary holding the accuracies for different values of k
    k_to_accuracies = {}

    # Perform k-fold cros validation to find the best value of k
    for k in k_choices:
        for j in range(num_folds):
            all_but_one_ind = [i for i in range(num_folds) if i != j]
            X_all_but_one = np.concatenate(X_train_folds[all_but_one_ind])
            y_all_but_one = np.concatenate(y_train_folds[all_but_one_ind])

            knn = kNearestNeighbor()
            knn.train(X_all_but_one, y_all_but_one)
            y_pred_k_f = knn.predict(X_train_folds[j], k)

            acc = float(sum(y_pred_k_f == y_train_folds[j])) / NUM_TEST

            if k not in k_to_accuracies:
                k_to_accuracies[k] = []
            k_to_accuracies[k].append(acc)

    # Print out the computed accuracies
    print('\n')
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    # Plot the raw obeservations
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

    # Based on the cross-validation results above, choose the best value for k,   
    # retrain the classifier using all the training data, and test it on the test
    # data.
    best_k = 1

    classifier = kNearestNeighbor()
    classifier.train(X_train, y_train)
    y_test_pred = classifier.predict(X_test, k=best_k)

    # Compute and display the accuracy
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / NUM_TEST
    print('\nGot %d / %d correct ==> accuracy: %f' % (num_correct, NUM_TEST, accuracy))

    return 0

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
