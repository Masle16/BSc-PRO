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

    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 

    areas = [cv2.contourArea(cnt) for cnt in cnts]

    return cnts, areas

def main():
    """ Main function """

    ################## IMPORT IMAGES ##################

    # Baggrund
    path = str(Path('images_1280x720/baggrund/bevægelse/*.jpg').resolve())
    background_fil = glob.glob(path)
    background_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in background_fil]

    # Ketchup
    path = str(Path('dataset2/images/kethchup/*.jpg').resolve())
    ketchup_fil = glob.glob(path)
    ketchup_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in ketchup_fil]

    ################## CREATE BACKGROUND MASK ##################

    # Initialize the mask
    mask = np.ones(shape=ketchup_images[0].shape, dtype='uint8') * 255

    # BACKGROUND IMAGE
    cnts_bgd, areas_bgd = get_contours(background_images[0], 90)

    index = np.argmax(areas_bgd)
    cv2.drawContours(mask, cnts_bgd, index, 0, -1)
    areas_bgd.pop(index)
    cnts_bgd.pop(index)

    index = np.argmax(areas_bgd)
    cv2.drawContours(mask, cnts_bgd, index, 0, -1)
    areas_bgd.pop(index)
    cnts_bgd.pop(index)

    # KETCHUP IMAGE
    cnts_ketch, areas_ketch = get_contours(ketchup_images[0], 125)

    index = np.argmax(areas_ketch)
    cv2.drawContours(mask, cnts_ketch, index, 0, -1)
    areas_ketch.pop(index)
    cnts_ketch.pop(index)

    index = np.argmax(areas_ketch)
    # cv2.drawContours(mask, cnts_ketch, index, 0, -1)
    areas_ketch.pop(index)
    cnts_ketch.pop(index)

    index = np.argmax(areas_ketch)
    cv2.drawContours(mask, cnts_ketch, index, 0, -1)
    areas_ketch.pop(index)
    cnts_ketch.pop(index)

    # Bitwise add mask to img
    img = cv2.bitwise_and(ketchup_images[0], mask)

    cv2.imshow('Mask', mask)
    cv2.imshow('Image', img)

    path = str(Path('preprocessing/bgd_mask.jpg').resolve())
    cv2.imwrite(path, mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()
