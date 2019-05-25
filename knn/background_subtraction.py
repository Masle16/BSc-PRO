"""
Module for background subtration
"""

###### IMPORTS ######
import glob
from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt

###### GLOBAL VARIABLES ######

###### FUNCTIONS ######
def show_img(img, window_name, width=352, height=240, wait_key=False):
    """ Show image in certain size """

    resized = cv2.resize(img,
                         (width, height),
                         interpolation=cv2.INTER_NEAREST)

    cv2.imshow(window_name, resized)

    if wait_key is True:
        cv2.waitKey(0)

    return 0

def background_sub(img, bgd, bgd_mask):
    """ Returns cropped image(448 x 448) of region of interest """

    ################## CALCULATE DIFFERENCE ##################
    _img = img.copy()
    _img = cv2.bitwise_and(_img, _img, mask=bgd_mask)
    diff = cv2.absdiff(bgd, _img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(diff_gray, 60, 255, 0)

    ################## REMOVE NOISE ##################
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CROSS, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.bitwise_and(thresh, bgd_mask)

    ################## FIND CONTOURS ##################
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

def background_sub2(img, avg_bgd, bgd_mask):
    """
    Performs background subtraction\n
    Returns region of interest(448 x 448) and bounding rect of found contour\n
    @img is the input image\n
    @avg_bgd is the average background\n
    @bgd_mask is the mask to remove unnessary background
    """

    ################## CALCULATE DIFFERENCE ##################
    _img = img.copy()
    _img = cv2.bitwise_and(_img, _img, mask=bgd_mask)
    diff = cv2.absdiff(avg_bgd, _img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(diff_gray, 60, 255, 0)

    ################## REMOVE NOISE ##################
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.bitwise_and(thresh, bgd_mask)

    ################## FIND CONTOURS ##################
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 1550]

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

    ################## BACKGROUND SUBTRACTION ##################

    # Background mask
    path = str(Path('preprocessing/bgd_mask.jpg').resolve())
    bgd_mask = cv2.imread(path, cv2.IMREAD_COLOR)
    bgd_mask_gray = cv2.cvtColor(bgd_mask, cv2.COLOR_BGR2GRAY)

    # Average background image
    path = str(Path('preprocessing/avg_background.jpg').resolve())
    background_img = cv2.imread(path, cv2.IMREAD_COLOR)

    img = cv2.bitwise_and(img, bgd_mask)
    regions, _ = background_sub2(img, background_img, bgd_mask_gray)
    for j, region in enumerate(regions):
        (x_left, x_right, y_up, y_down) = region
        roi = img[y_up : y_down, x_left : x_right]

    return 0

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
