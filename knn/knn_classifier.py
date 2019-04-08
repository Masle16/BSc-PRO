#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

""" Module for k Nearest Neighbour """

###### IMPORTS ######
import os
import glob
import random
from pathlib import Path
from imutils import paths
import cv2
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from background_subtraction import background_sub2

###### GLOBAL VARIABLES ######

###### FUNCTIONS ######
def ocr_of_hand_written_digits():
    """
    OCR of hand-written digits
    """

    path = str(Path('knn/digits.png').resolve())
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # Make it into a numpy array. It size will be (50, 100, 20, 20)
    x = np.array(cells)

    # Now we prepate train_data and test_data
    train = x[:, :50].reshape(-1, 400).astype(np.float32)   # size = (2500, 400)
    test = x[:, 50:100].reshape(-1, 400).astype(np.float32) # size = (2550, 400)

    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k, 250)[:, np.newaxis]
    test_labels = train_labels.copy()

    # Initiate Knn, train the data, then test it with test data for k=1
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    ret, result, neighbours, dist = knn.findNearest(test, k=5)

    # Now we check the accuracy of classification
    # for that, compare the result with test_labels and check which are wrong
    matches = (result == test_labels)
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / result.size
    print(accuracy)

def ocr_of_english_aplhabets():
    """
    OCR for english alphabet
    """
    # Load the data, converters convert the letter to a number
    filename = str(Path('knn/letter-recognition.data.txt').resolve())
    data = np.loadtxt(filename,
                      dtype='float32',
                      delimiter=',',
                      converters={0: lambda ch: ord(ch)-ord('A')})

    # split the data to two, 10000 each for train and test
    train, test = np.vsplit(data, 2)

    # split trainData and testData to features and responses
    responses, trainData = np.hsplit(train, [1])
    labels, testData = np.hsplit(test, [1])

    # Initiate the kNN, classify, measure accuracy.
    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    ret, result, neighbours, dist = knn.findNearest(testData, k=5)
    correct = np.count_nonzero(result == labels)
    accuracy = correct * 100.0 / 10000
    print(accuracy)

def image_to_feature_vector(image, size=(32, 32)):
    """
    Rezise the image to a fixed size, then flatten the image into
    a list of raw pixel intensities.
    """

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

def knn_classifier():
    """
    """

    # Grab the list of images that we will be describing
    path = '/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/knn/cropped/'
    image_paths = list(paths.list_images(path))

    # Initialize the raw pixel intensities matrix, the features matrix,
    # and labels list
    # raw_images = []
    features = []
    labels = []

    # loop over the input images
    for (i, image_path) in enumerate(image_paths):
        # Load the image and extract the class label
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = image_path.split(os.path.sep)[-1].split('.')[0]

        # # Extract raw pixel intensity "features"
        # pixels = image_to_feature_vector(image)

        # Extract color histogram
        hist = extract_color_histogram(image)

        # Update the raw images, features and labels matricies respectively
        # raw_images.append(pixels)
        features.append(hist)
        labels.append(label)

        # Show an update every 100 images
        if i > 0 and i % 100 == 0:
            print('[INFO] processed {}/{}'.format(i, len(image_paths)))

    # Show some information on the memory consumed by the raw images
    # matrix and features matrix
    # raw_images = np.array(raw_images)
    features = np.array(features)
    labels = np.array(labels)
    # print('[INFO] pixels matrix: {:.2f}MB'.format(raw_images.nbytes / (1024 * 1000.0)))
    print('[INFO] features matrix: {:.2f}MB'.format(features.nbytes / (1024 * 1000.0)))

    # Partition the data into training and testing splits, using 75 %
    # of the data for training and the remaing 25 % for testing
    # (train_ri, test_ri, train_rl, test_rl) = train_test_split(
    #     raw_images,
    #     labels,
    #     test_size=0.25,
    #     random_state=42
    # )
    (train_feat, test_feat, train_labels, test_labels) = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42
    )

    for i in range(11):
        if i == 0:
            continue

        # # Train and evaluate a kNN classifier on the raw pixel intensities
        # print('[INFO] evaluating raw pixels accuracy...')
        # model = KNeighborsClassifier(n_neighbors=i, n_jobs=1)
        # model.fit(train_ri, train_rl)
        # acc = model.score(test_ri, test_rl)
        # print('[INFO] raw pixel accuracy: {:.2f}%'.format(acc * 100))

        # Train an evaluate a kNN classifier on the histogram representations
        print('[INFO] k =', str(i))
        print('[INFO] evaluating histogram accuracy...')
        model = KNeighborsClassifier(n_neighbors=i, n_jobs=1)
        model.fit(train_feat, train_labels)
        acc = model.score(test_feat, test_labels)
        print('[INFO] histogram accuracy: {:.2f}%'.format(acc * 100))

###### MAIN FUNCTION ######
def main():
    """ Main function """

    # Import images
    # paths = [
    #     str(Path('dataset3/res_still/train/background/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/train/potato/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/train/carrots/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/train/catfood_salmon/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/train/catfood_beef/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/train/bun/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/train/arm/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/train/ketchup/*.jpg').resolve()),
    #     str(Path('dataset3/images/All/*.jpg').resolve())
    # ]

    # path = str(Path('knn/bgd_mask.jpg').resolve())
    # bgd_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # path = str(Path('knn/avg_background.jpg').resolve())
    # avg_bgd = cv2.imread(path, cv2.IMREAD_COLOR)

    # path = str(Path('dataset3/res_still/train/ketchup/*.jpg').resolve())
    # filenames = glob.glob(path)
    # images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in filenames]
    # for i, img in enumerate(images):
    #     regions, cnts = background_sub2(img, avg_bgd, bgd_mask)
    #     for region in regions:
    #         (x_left, x_right, y_up, y_down) = region
    #         roi = img[y_up : y_down, x_left : x_right]

    #         path = str(Path('knn/dataset/ketchup.' + str(i) + '.jpg').resolve())
    #         cv2.imwrite(path, roi)

    # path = '/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/knn/dataset/'
    # image_paths = list(paths.list_images(path))
    # for (i, img_path) in enumerate(image_paths):
    #     image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #     label = img_path.split(os.path.sep)[-1].split('.')[0]

    #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     lower = (0, 70, 0)
    #     upper = (179, 255, 255)
    #     mask = cv2.inRange(src=hsv, lowerb=lower, upperb=upper)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #     img = cv2.bitwise_and(image, image, mask=mask)

    #     path = str(Path('knn/cropped/' + label + '.' + str(i) + '.jpg').resolve())
    #     cv2.imwrite(path, img)

    knn_classifier()

    return 0

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
