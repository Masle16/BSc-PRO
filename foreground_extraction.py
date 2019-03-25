#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

""" Module for foreground extraction """

###### IMPORTS ######
import glob
import random
from pathlib import Path
import cv2
import numpy as np

from preprocessing.background_subtration import background_subtration as bs

###### GLOBAL VARIABLES ######
BGD_MASK = cv2.imread(str(Path('preprocessing/background_mask.jpg').resolve()),
                      cv2.IMREAD_GRAYSCALE)

###### FUNCTIONS ######
def foreground_extraction(src, bgd_img):
    """ Foreground extration function """

    # Get region of interest
    (x_left, x_right, y_up, y_down), (x, y, width, height) = bs.background_sub(img=src,
                                                                               bgd=bgd_img,
                                                                               bgd_mask=BGD_MASK)
    rect = (x, y, width, height)

    # Grab cut
    mask = np.zeros(src.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img=src,
                mask=mask,
                rect=rect,
                bgdModel=bgd_model,
                fgdModel=fgd_model,
                iterCount=5,
                mode=cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    img = src * mask[:, :, np.newaxis]
    img = img[y_up : y_down, x_left : x_right]

    return img

###### MAIN FUNCTION ######
def main():
    """ Main function """

    ####### IMPORT IMAGES #######

    # Baggrund
    path = str(Path('images_1280x720/baggrund/bevægelse/*.jpg').resolve())
    background_fil = glob.glob(path)
    background_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in background_fil]

    # Generate average background image
    background_img = bs.run_avg(background_images)

    # Guleroedder
    path = str(Path('images_1280x720/gulerod/still/*.jpg').resolve())
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

    input_images = carrot_images + potato_images + cat_sal_images + cat_beef_images
    random.shuffle(input_images)
    random.shuffle(input_images)
    random.shuffle(input_images)

    ####### FOREGROUND EXTRACTION #######

    for input_img in input_images:
        img = foreground_extraction(input_img, background_img)

        cv2.imshow('Foreground', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
