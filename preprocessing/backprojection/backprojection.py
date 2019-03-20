#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

"""
Module for Back-projection
"""

import glob
from pathlib import Path
import random
import cv2
import numpy as np
# from matplotlib import pyplot as plt

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

def remove_background():
    """ returns image with no background, only table """

    # Get background
    path = '/home/mikkel/Documents/github/BSc-PRO/images_1280x720/baggrund/bevægelse/WIN_20190131_10_31_36_Pro.jpg'
    background = cv2.imread(path, cv2.IMREAD_COLOR)

    # Find background pixels coordinates
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 125), (179, 100, 255))
    result = cv2.bitwise_and(background, background, mask=mask)

    return mask, result

def backproject(roi_hist, img, background_mask):
    """
    returns backprojected image
    @roi_hist, histogram of region of interest to find
    @img, image to search in
    """

    # Remove unessesary background
    _img = cv2.bitwise_and(img, img, mask=background_mask)
    _img = cv2.blur(_img, (5, 5))

    # Convert to HSV
    img_hsv = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)

    # Create Histogram of roi and create mask from the histogram
    mask = cv2.calcBackProject([img_hsv], [0, 1], roi_hist, [0, 180, 0, 256], scale=1)

    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.filter2D(mask, -1, kernel)
    _, mask = cv2.threshold(mask, 127, 200, cv2.THRESH_BINARY)

    mask = cv2.merge((mask, mask, mask))
    result = cv2.bitwise_and(_img, mask)

    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    # Find contours
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 0, 127, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    cnt_img = img.copy()
    for cnt in contours:
        cv2.drawContours(cnt_img, cnt, -1, random_color(), 3)

    show_img(cnt_img, 'Contours')

    # Find biggest contour
    cnts = []
    areas = [cv2.contourArea(c) for c in contours]

    k = 3
    for _ in range(k):
        index = areas.index(max(areas))
        cnts.append(contours[index])
        areas[index] = 0.0

    # for i, area in enumerate(areas):
    #     if area >= 50.0:
    #         cnts.append(contours[i])

    img_crop = []
    img_rect = img.copy()
    for cnt in cnts:
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

        img_crop.append(img[y_up : y_down, x_left : x_right])

        cv2.rectangle(img_rect, (x_left, y_up), (x_right, y_down), random_color(), 4)

    show_img(img_rect, 'Region of interest', wait_key=True)

    return img_crop

def main():
    """ Main function """

    ################## IMPORT IMAGES ##################

    # Baggrund
    path = str(Path('images_1280x720/baggrund/bevægelse/*.jpg').resolve())
    background_fil = glob.glob(path)
    background_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in background_fil]

    # Guleroedder
    path = str(Path('images_1280x720/gulerod/still/*.jpg'))
    carrot_fil = glob.glob(path)
    carrot_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in carrot_fil]

    # Kartofler
    path = str(Path('images_1280x720/kartofler/still/*.jpg').resolve())
    potato_fil = glob.glob(path)
    potato_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in potato_fil]

    # Kat laks
    path = str(Path('images_1280x720/kat_laks/still/*.jpg').resolve())
    cat_sal_fil = glob.glob(path)
    cat_sal_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_sal_fil]

    # Kat okse
    path = str(Path('images_1280x720/kat_okse/still/*.jpg').resolve())
    cat_beef_fil = glob.glob(path)
    cat_beef_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_beef_fil]

    ################## Back-projection ##################

    # Create template histogram
    path = str(Path('preprocessing/backprojection/template_sal_potato.jpg').resolve())
    roi_img = cv2.imread(path, cv2.IMREAD_COLOR)
    roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Import background mask
    path = str(Path('preprocessing/background_mask.jpg').resolve())
    background_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    for img in cat_sal_images:
        roi = backproject(roi_hist, img, background_mask)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
