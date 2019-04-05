#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

"""
Module for color picker
"""

from pathlib import Path
import cv2
import numpy as np

def nothing(args):
    """ nothing """
    pass

def main():
    """ Main function """

    # Create a black image, a window
    img_path = str(Path('dataset3/images/All/WIN_20190322_09_48_11_Pro.jpg').resolve())
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    cv2.namedWindow('Trackbars')

    # create trackbars for color change
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    while True:

        k = cv2.waitKey(1)
        if k == 27:
            break

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower_blue = np.array([l_h, l_s, l_v])
        upper_blue = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        result = cv2.bitwise_and(img, img, mask=mask)

        # cv2.imshow("frame", img)
        # cv2.imshow("mask", mask)
        cv2.imshow("result", result)

    # path = str(Path('preprocessing/background_mask.jpg').resolve())
    # cv2.imwrite(path, mask)

    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
