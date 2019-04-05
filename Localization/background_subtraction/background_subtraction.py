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

###### FUNCTIONS ######
def random_color():
    """ Generate random color """
    rgbl = [255, 0, 0]
    random.shuffle(rgbl)

    return tuple(rgbl)

def show_img(img, window_name, width=352, height=240, wait_key=False):
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

    ################## CALCULATE DIFFERENCE ##################
    _img = img.copy()
    _img = cv2.bitwise_and(_img, _img, mask=bgd_mask)
    diff = cv2.absdiff(bgd, _img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(diff_gray, 50, 255, 0)

    ################## REMOVE NOISE ##################
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CROSS, kernel)
    thresh = cv2.bitwise_and(thresh, bgd_mask)

    ################## FIND CONTOURS ##################
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]

    if not areas:
        rows, cols = img.shape[:2]
        x = int(cols / 2)
        y = int(rows / 2)
        width = 50
        height = 50
    else:
        index = np.argmax(areas)
        cnt = contours[index]
        x, y, width, height = cv2.boundingRect(cnt)

    x_ctr = int((x + (x + width)) / 2)
    y_ctr = int((y + (y + height)) / 2)
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

    # Return region (448 x 448) and bounding rect of found contour
    return (x_left, x_right, y_up, y_down), (x, y, width, height)

def background_sub2(img, bgd, bgd_mask):
    """
    Performs background subtraction\n
    Returns region of interest(448 x 448) and bounding rect of found contour\n
    @img is the input image\n
    @bgd is the average background\n
    @bgd_mask is the mask to remove unnessary background
    """

    ################## CALCULATE DIFFERENCE ##################
    _img = img.copy()
    _img = cv2.bitwise_and(_img, _img, mask=bgd_mask)
    diff = cv2.absdiff(bgd, _img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(diff_gray, 60, 255, 0)

    ################## REMOVE NOISE ##################
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.bitwise_and(thresh, bgd_mask)

    ################## FIND CONTOURS ##################
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    cnts = []
    if not contours:
        # If nothing found return center of image
        rows, cols = img.shape[:2]
        x = int(cols / 2)
        y = int(rows / 2)
        width = 50
        height = 50

        cnt = (x, y, width, height)
        cnts.append(cnt)

        # Find region 448 x 448
        x_ctr = int((x + (x + width)) / 2)
        y_ctr = int((y + (y + height)) / 2)
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

        region = (x_left, x_right, y_up, y_down)
        regions.append(region)
    else:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        print(areas)

        for i, area in enumerate(areas):
            if area < 2200:
                continue

            # Find bounding rect of contour
            contour = contours[i]
            x, y, width, height = cv2.boundingRect(contour)
            cnt = (x, y, width, height)
            cnts.append(cnt)

            # Find region 448 x 448
            x_ctr = int((x + (x + width)) / 2)
            y_ctr = int((y + (y + height)) / 2)
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

            region = (x_left, x_right, y_up, y_down)
            regions.append(region)

    return regions, cnts

def main():
    """ Main function """

    # ################## IMPORT IMAGES ##################
    # path_images = [
    #     str(Path('dataset3/res_still/test/background/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/test/potato/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/test/carrots/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/test/catfood_salmon/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/test/catfood_beef/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/test/bun/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/test/arm/*.jpg').resolve()),
    #     str(Path('dataset3/res_still/test/kethchup/*.jpg').resolve())
    # ]

    # ################## BACKGROUND SUBTRACTION ##################

    # # Background mask
    # path = str(Path('preprocessing/bgd_mask.jpg').resolve())
    # mask = cv2.imread(path, cv2.IMREAD_COLOR)
    # mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # # Average background image
    # path = str(Path('preprocessing/avg_background.jpg').resolve())
    # background_img = cv2.imread(path, cv2.IMREAD_COLOR)

    # path = str(Path('dataset3/images/All/*.jpg').resolve())

    # images_fil = glob.glob(path_images[3])
    # images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in images_fil]

    # for img in images:
    #     regions, cnts = background_sub2(img, background_img, mask_gray)

    #     for region in regions:
    #         for cnt in cnts:
    #             (x_left, x_right, y_up, y_down) = region
    #             (x, y, width, height) = cnt

    #             # cv2.rectangle(img=img,
    #             #               pt1=(x_left, y_up),
    #             #               pt2=(x_right, y_down),
    #             #               color=(255, 0, 0),
    #             #               thickness=3)

    #             cv2.rectangle(img=img,
    #                           pt1=(x, y),
    #                           pt2=(x + width, y + height),
    #                           color=(0, 0, 255),
    #                           thickness=3)

    #     cv2.imshow('Regions of interest', img)
    #     cv2.waitKey(0)

    return 0

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
