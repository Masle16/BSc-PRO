#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

""" Module for background subtration """

###### IMPORTS ######
import glob
from pathlib import Path
import imutils
import cv2
import numpy as np

###### GLOBAL VARIABLES ######

###### FUNCTIONS ######
def get_contours(img, th):
    """ Returns contours """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    areas = [cv2.contourArea(cnt) for cnt in cnts]

    return cnts, areas

def main():
    """ Main function """

    ################## IMPORT IMAGES ##################
    # All
    path = str(Path('dataset3/res_still/test/catfood_salmon/WIN_20190131_11_26_46_Pro.jpg').resolve())
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    ################## CREATE BACKGROUND MASK ##################

    # Initialize the mask
    mask = np.ones(shape=image.shape, dtype='uint8') * 255

    path = str(Path('preprocessing/bgd_mask.jpg').resolve())
    bgd_mask = cv2.imread(path, cv2.IMREAD_COLOR)

    # IMAGE
    cnts, areas = get_contours(image, 130)

    print(areas)

    for cnt in cnts:
        cv2.drawContours(mask, [cnt], -1, 0, -1)

    for i, area in enumerate(areas):
        if area > 10 and area < 200:
            cv2.drawContours(bgd_mask, cnts, i, 0, -1)

    # Bitwise add mask to img
    img = cv2.bitwise_and(image, bgd_mask)

    cv2.imshow('Mask', mask)
    cv2.imshow('bgd mask', bgd_mask)

    # path = str(Path('preprocessing/bgd_mask2.jpg').resolve())
    # cv2.imwrite(path, bgd_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()
