#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

"""
Module for background subtration
"""

###### IMPORTS ######
import glob
import random
from pathlib import Path
import cv2
import numpy as np
#from matplotlib import pyplot as plt

###### GLOBAL VARIABLES ######
NUMBER = 0

def random_color():
    """ Generate random color """
    rgbl = [255, 0, 0]
    random.shuffle(rgbl)

    return tuple(rgbl)

def show_img(img, window_name, width=640, height=480, wait_key=False):
    """ Show image in certain size """

    resized = cv2.resize(img,
                         (width, height),
                         interpolation=cv2.INTER_NEAREST)

    cv2.imshow(window_name, resized)

    if wait_key is True:
        cv2.waitKey(0)

    return 0

def remove_background(img):
    """ Returns image with no background, only table """

    # Find background pixels coordinates
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 64), (179, 51, 255))
    result = cv2.bitwise_and(img, img, mask=mask)

    return mask, result

def run_avg(background_images):
    """ Returns running average of all images in path folder """

    avg = np.float32(background_images[0])

    for img in background_images:
        cv2.accumulateWeighted(img, avg, 0.1)

    result = cv2.convertScaleAbs(avg)

    return result

def background_sub(img, bgd, bgd_mask):
    """ Returns cropped image(448 x 448) of region of interest """

    # Create copy to work on
    _img = img.copy()

    # Calculate image difference and find largest contour
    diff = cv2.absdiff(bgd, _img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Remove unessesary background
    diff_gray = cv2.bitwise_and(diff_gray, diff_gray, mask=bgd_mask)

    # Remove small differences
    _, thresh = cv2.threshold(diff_gray, 25, 255, 0)

    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 4))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Get contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate contours pixel intensity
    cnt_pixel_value = []
    for contour in contours:
        pixel_sum = 0
        contour = np.asarray(contour)
        contour = contour.reshape(contour.shape[0], contour.shape[2])
        pixel_sum = diff_gray[contour[:, :][:, 1], contour[:, :][:, 0]]
        cnt_pixel_value.append(np.sum(pixel_sum))

    # Selected contour with highest pixel intensity
    index = np.argmax(cnt_pixel_value)
    cnt = contours[index]

    # Crop contour form image
    _x, _y, _w, _h = cv2.boundingRect(cnt)
    x_ctr = int((_x + (_x + _w)) / 2)
    y_ctr = int((_y + (_y + _h)) / 2)
    radius = 224
    x_left = x_ctr - radius
    x_right = x_ctr + radius
    y_up = y_ctr - radius
    y_down = y_ctr + radius

    if x_right > img.shape[1]:
        margin = -1 * (img.shape[1] - x_right)
        x_right -= margin
        x_left -= margin
    elif x_left < 0:
        margin = -1 * x_left
        x_right += margin
        x_left += margin

    if y_up < 0:
        margin = -1 * y_up
        y_down += margin
        y_up += margin
    elif y_down > img.shape[0]:
        margin = -1 * (img.shape[0] - y_down)
        y_down -= margin
        y_up -= margin

    return (x_left, x_right, y_up, y_down), (_x, _y, _w, _h)

def main():
    """ main function """

    ################## IMPORT IMAGES ##################

    # Baggrund
    path = str(Path('dataset2/images/baggrund/*.jpg').resolve())
    background_fil = glob.glob(path)
    background_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in background_fil]

    # Guleroedder
    path = str(Path('dataset2/images/carrots/*.jpg'))
    carrot_fil = glob.glob(path)
    carrot_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in carrot_fil]

    # Kartofler
    path = str(Path('dataset2/images/potato/*.jpg').resolve())
    potato_fil = glob.glob(path)
    potato_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in potato_fil]

    # Kat laks
    path = str(Path('dataset2/images/catfood_salmon/*.jpg').resolve())
    cat_sal_fil = glob.glob(path)
    cat_sal_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_sal_fil]

    # Kat okse
    path = str(Path('dataset2/images/catfood_beef/*.jpg').resolve())
    cat_beef_fil = glob.glob(path)
    cat_beef_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_beef_fil]

    # Boller
    path = str(Path('dataset2/images/bun/*.jpg').resolve())
    bun_fil = glob.glob(path)
    bun_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in bun_fil]

    # Arm
    path = str(Path('dataset2/images/arm/*.jpg').resolve())
    arm_fil = glob.glob(path)
    arm_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in arm_fil]

    # Ketchup
    path = str(Path('dataset2/images/kethchup/*.jpg').resolve())
    ketchup_fil = glob.glob(path)
    ketchup_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in ketchup_fil]

    # Combine images
    input_images = (background_images +
                    carrot_images +
                    potato_images +
                    cat_sal_images +
                    cat_beef_images +
                    bun_images +
                    arm_images +
                    ketchup_images)

    # Shuffle
    random.shuffle(input_images)
    random.shuffle(input_images)
    random.shuffle(input_images)
    random.shuffle(input_images)

    ################## BACKGROUND SUBTRACTION ##################

    # Background mask
    path = str(Path('preprocessing/bgd_mask_1.jpg').resolve())
    bgd_mask_1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    path = str(Path('preprocessing/bgd_mask_2.jpg').resolve())
    bgd_mask_2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.bitwise_and(bgd_mask_1, bgd_mask_1, mask=bgd_mask_2)

    # Create average background image and remove unnessary background
    background_img = run_avg(background_images)
    background_img = cv2.bitwise_and(background_img, background_img, mask=mask)

    for img in input_images:
        roi, coordinates = background_sub(img, background_img, mask)

        (x_left, x_right, y_up, y_down) = roi
        (x, y, width, height) = coordinates

        cv2.rectangle(img=img,
                      pt1=(x_left, y_up),
                      pt2=(x_right, y_down),
                      color=(255, 0, 0),
                      thickness=3)

        cv2.rectangle(img=img,
                      pt1=(x, y),
                      pt2=(x + width, y + height),
                      color=(0, 0, 255),
                      thickness=3)

        cv2.imshow('Image', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()
