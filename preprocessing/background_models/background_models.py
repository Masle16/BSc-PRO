import cv2
import numpy as np
from matplotlib import pyplot as plt

def diff(src, background):
    """
    returns region of interest when background is subtracted from img
    @img, image with objects
    @background, static background with objects which is not interesting
    """

    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(background, img)

    retval, thresh = cv2.threshold(diff, 250, 255, cv2.THRESH_BINARY)

    cv2.imshow("diff", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

src = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/potato_and_catfood/train/potato/WIN_20190131_09_59_39_Pro.jpg')
background = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/background_models/background.jpg')

diff(src, background)