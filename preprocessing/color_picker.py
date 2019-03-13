""" Module for color picker """

import cv2
import numpy as np

def nothing(args):
    """ nothing """
    pass

def main():
    """ Main function """

    # Create a black image, a window
    img = cv2.imread('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/\
images_1280x720/baggrund/bevægelse/WIN_20190131_10_31_36_Pro.jpg', cv2.IMREAD_COLOR)
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

        img = cv2.imread('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/\
images_1280x720/baggrund/bevægelse/WIN_20190131_10_31_36_Pro.jpg', cv2.IMREAD_COLOR)
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

        cv2.imshow("frame", img)
        cv2.imshow("mask", mask)
        cv2.imshow("result", result)

    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
