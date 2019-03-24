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
    img_path = str(Path('images_1280x720/kat_okse/still/WIN_20190131_11_31_31_Pro.jpg').resolve())
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    cv2.namedWindow('Trackbars')

    # create trackbars for color change
    cv2.createTrackbar("Low", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("High", "Trackbars", 255, 255, nothing)

    while True:

        k = cv2.waitKey(1)
        if k == 27:
            break

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        low = cv2.getTrackbarPos("Low", "Trackbars")
        high = cv2.getTrackbarPos("High", "Trackbars")

        _, thresh = cv2.threshold(gray, low, high, cv2.THRESH_BINARY)
        cv2.imshow("result", thresh)

    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
