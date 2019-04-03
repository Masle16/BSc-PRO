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
def simpleFilter(src, bgd_mask):
    """
    \n
        @param src is the input image\n
        @param bgd_mask is the mask to remove the top corners
    """

    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lower = (0, 70, 0)
    upper = (179, 255, 255)
    mask = cv2.inRange(src=hsv, lowerb=lower, upperb=upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.bitwise_and(mask, bgd_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        rows, cols = src.shape[:2]
        x = int(cols / 2)
        y = int(rows / 2)
        width = height = 50
    else:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        index = np.argmax(areas)
        cnt = contours[index]
        x, y, width, height = cv2.boundingRect(cnt)

    return (x, y, width, height)

###### MAIN FUNCTION ######
def main():
    """ Main function """

    ####### IMPORT IMAGES #######
    path_images = [
        str(Path('dataset2/images/baggrund/*.jpg').resolve()),
        str(Path('dataset2/images/potato/*.jpg').resolve()),
        str(Path('dataset2/images/carrots/*.jpg').resolve()),
        str(Path('dataset2/images/catfood_salmon/*.jpg').resolve()),
        str(Path('dataset2/images/catfood_beef/*.jpg').resolve()),
        str(Path('dataset2/images/bun/*.jpg').resolve()),
        str(Path('dataset2/images/arm/*.jpg').resolve()),
        str(Path('dataset2/images/kethchup/*.jpg').resolve())
    ]

    ####### IMPORT BACKGROUND MASK #######
    path = str(Path('template_matching/bgd_mask.jpg').resolve())
    bgd_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    images_fil = glob.glob(path_images[2])
    images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in images_fil]

    for src in images:
        cnt = simpleFilter(src, bgd_mask)
        (x, y, width, height) = cnt

        cv2.rectangle(img=src,
                      pt1=(x, y),
                      pt2=(x + width, y + height),
                      color=(255, 0, 0),
                      thickness=3)

        cv2.imshow('Image', src)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
