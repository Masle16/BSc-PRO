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

def main():
    """ Main function """

    ################## IMPORT IMAGES ##################

    # Baggrund
    path = str(Path('images_1280x720/baggrund/bevægelse/*.jpg').resolve())
    background_fil = glob.glob(path)
    background_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in background_fil]

    ################## CREATE BACKGROUND MASK ##################

    img = background_images[0].copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)

    # # Find edges in image
    # edged = cv2.Canny(background_mask, 25, 200)

    # Find contours in the image
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the mask
    mask = np.ones(img.shape[:2], dtype='uint8') * 255

    # Loop over the contours
    for cnt in cnts:
        cv2.drawContours(mask, [cnt], 0, 0, -1)

    # Remove contours
    img = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('Mask', mask)
    cv2.imshow('Image', img)

    path = str(Path('preprocessing/bgd_mask_2.jpg').resolve())
    cv2.imwrite(path, mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()
