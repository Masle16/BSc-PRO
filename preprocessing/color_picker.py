""" Module for color picker """

import cv2
# import numpy as np

def nothing():
    """ nothing """
    pass

def main():
    """ Main function """

    # Create a black image, a window
    img = cv2.imread('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/\
images_1280x720/baggrund/bevægelse/WIN_20190131_10_31_36_Pro.jpg', cv2.IMREAD_COLOR)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('H', 'image', 0, 255, nothing)
    cv2.createTrackbar('S', 'image', 0, 255, nothing)
    cv2.createTrackbar('V', 'image', 0, 255, nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)

    while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        _r = cv2.getTrackbarPos('R', 'image')
        _g = cv2.getTrackbarPos('G', 'image')
        _b = cv2.getTrackbarPos('B', 'image')
        _s = cv2.getTrackbarPos(switch, 'image')

        if _s == 0:
            img[:] = 0
        else:
            img[:] = [_b, _g, _r]

    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
