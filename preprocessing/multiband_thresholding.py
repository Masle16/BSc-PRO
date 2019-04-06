#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

"""
Module for template matching and chamfer matching
"""

###### IMPORTS ######
import glob
import random
from pathlib import Path
import cv2
import numpy as np

###### GLOBAL VARIABLES ######
DOWNSCALING = 4
CLASSES = ['Potato', 'Carrot', 'Cat beef', 'Cat salmon']
# BACKGROUND = 0
POTATO = 0
CARROT = 1
CAT_SAL = 2
CAT_BEEF = 3
BUN = 4
ARM = 5
KETCHUP = 6

###### FUNCTIONS ######
def show_img(img, window_name, width=352, height=240, wait_key=False):
    """
    Show image in certain size
    """

    resized = cv2.resize(img,
                         (width, height),
                         interpolation=cv2.INTER_NEAREST)

    cv2.imshow(window_name, resized)

    if wait_key is True:
        cv2.waitKey(0)

    return 0

def multibandThresholding(src, bgd_mask):
    """
    returns \n
        @param src is the input image\n
        @param bgd_mask is the mask to remove the top corners
    """

    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lower = (0, 70, 0)
    upper = (179, 255, 255)
    mask = cv2.inRange(src=hsv, lowerb=lower, upperb=upper)
    show_img(mask, 'InRange')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.bitwise_and(mask, bgd_mask)
    show_img(mask, 'Mask')

    ###### FIND CONTOURS ######
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    cnts = []
    if not contours:
        # Contour
        rows, cols = src.shape[:2]
        x = int(cols / 2)
        y = int(rows / 2)
        width = height = 50
        cnt = (x, y, width, height)
        cnts.append(cnt)

        # Region
        x_ctr = int((x + (x + width)) / 2)
        y_ctr = int((y + (y + height)) / 2)
        radius = 224
        x_left = x_ctr - radius
        x_right = x_ctr + radius
        y_up = y_ctr - radius
        y_down = y_ctr + radius

        if x_right > src.shape[1]:
            margin = -1 * (src.shape[1] - x_right)
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
        elif y_down > src.shape[0]:
            margin = -1 * (src.shape[0] - y_down)
            y_down -= margin
            y_up -= margin

        region = (x_left, x_right, y_up, y_down)
        regions.append(region)

    else:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        print(areas)

        for i, area in enumerate(areas):
            if area < 2000:
                continue

            # Contour
            cnt = contours[i]
            x, y, width, height = cv2.boundingRect(cnt)
            cnt = (x, y, width, height)
            cnts.append(cnt)

            # Region
            x_ctr = int((x + (x + width)) / 2)
            y_ctr = int((y + (y + height)) / 2)
            radius = 224
            x_left = x_ctr - radius
            x_right = x_ctr + radius
            y_up = y_ctr - radius
            y_down = y_ctr + radius

            if x_right > src.shape[1]:
                margin = -1 * (src.shape[1] - x_right)
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
            elif y_down > src.shape[0]:
                margin = -1 * (src.shape[0] - y_down)
                y_down -= margin
                y_up -= margin

            region = (x_left, x_right, y_up, y_down)
            regions.append(region)

    return regions, cnts

###### MAIN FUNCTION ######
def main():
    """ Main function """

    ####### IMPORT IMAGES #######
    path_images = [
        # str(Path('dataset3/res_still/test/background/*.jpg').resolve()),
        # str(Path('dataset3/res_still/test/potato/*.jpg').resolve()),
        # str(Path('dataset3/res_still/test/carrots/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/catfood_salmon/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/catfood_beef/*.jpg').resolve())
        # str(Path('dataset3/res_still/test/bun/*.jpg').resolve()),
        # str(Path('dataset3/res_still/test/arm/*.jpg').resolve()),
        # str(Path('dataset3/res_still/test/kethchup/*.jpg').resolve()),
        # str(Path('dataset3/images/All/*.jpg').resolve())
    ]

    ####### IMPORT BACKGROUND MASK #######
    path = str(Path('template_matching/bgd_mask.jpg').resolve())
    bgd_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    for path in path_images:

        images_fil = glob.glob(path)
        images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in images_fil]

        for src in images:
            regions, cnts = multibandThresholding(src=src, bgd_mask=bgd_mask)

            for region in regions:
                for cnt in cnts:
                    (x_left, x_right, y_up, y_down) = region
                    (x, y, width, height) = cnt

                    # cv2.rectangle(img=img,
                    #               pt1=(x_left, y_up),
                    #               pt2=(x_right, y_down),
                    #               color=(255, 0, 0),
                    #               thickness=3)

                    cv2.rectangle(img=src,
                                  pt1=(x, y),
                                  pt2=(x + width, y + height),
                                  color=(0, 0, 255),
                                  thickness=3)

            cv2.imshow('Image', src)
            cv2.waitKey(0)

    return 0

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
