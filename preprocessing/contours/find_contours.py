""" Module for finding image objection with contours detection """

import glob
import cv2
import numpy as np
#from matplotlib import pyplot as plt

WIDTH = 448
HEIGHT = 448

def show_img(img, window_name, width, height, wait_key=False):
    """ Show image in certain size """

    resized = cv2.resize(img,
                         (width, height),
                         interpolation=cv2.INTER_CUBIC)
    
    cv2.imshow(window_name, resized)

    if wait_key is True:
        cv2.waitKey(0)

    return 0

def find_contours(src):
    """

    Returns cropped image with region of interest
    @src, input image to find region of interest in

    """

    _hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    _mask = cv2.inRange(_hsv, (10, 100, 20), (20, 255, 200))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    
    # Opening (erosion followed by dilation)
    _mask = cv2.morphologyEx(_mask, cv2.MORPH_OPEN, kernel)

    # Closing (dilation followed by erosion)
    _mask = cv2.morphologyEx(_mask, cv2.MORPH_CLOSE, kernel)

    _canny_output = cv2.Canny(_mask, 25, 100)
    _cnts, _ = cv2.findContours(_canny_output,
                                cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)

    show_img(_mask, 'Mask', WIDTH, HEIGHT)
    show_img(_canny_output, 'Canny', WIDTH, HEIGHT)

    # # Draw contours
    # img = src.copy()
    # cv2.drawContours(img, _cnts, -1, (0, 255, 0), 3)
    # cv2.imshow("Contours", img)
    # cv2.waitKey(0)

    # Find biggest contour
    areas = [cv2.contourArea(c) for c in _cnts]
    max_index = np.argmax(areas)
    _cnt = _cnts[max_index]

    # Crop contour form image
    _x, _y, _w, _h = cv2.boundingRect(_cnt)
    x_ctr = int((_x + (_x + _w)) / 2)
    y_ctr = int((_y + (_y + _h)) / 2)
    radius = 224
    x_left, x_right, y_up, y_down = x_ctr - radius, x_ctr + radius, y_ctr - radius, y_ctr + radius

    if x_right > src.shape[1]:
        margin = -1 * (src.shape[1] - x_right)
        x_right -= margin
        x_left -= margin
    elif x_left < 0:
        margin = -1 * x_left
        x_right += margin
        x_left += margin

    if y_up < 0:
        margin = -1 * y_up
        y_down += margin
        y_up += margin
    elif y_down > src.shape[0]:
        margin = -1 * (src.shape[0] - y_down)
        y_down -= margin
        y_up -= margin

    img_crop = src[y_up : y_down, x_left : x_right]

    # cv2.imwrite('/home/mathi/Desktop/crop_img.jpg', img_crop)

    # Display detected area
    img_rect = src.copy()
    cv2.rectangle(img_rect, (x_left, y_up), (x_right, y_down), (0, 0, 255), 4)
    show_img(img_rect, 'Detected rect', WIDTH, HEIGHT)

    cv2.waitKey(0)

    return img_crop


def main():
    """ Main function """

    path = '/mnt/sdb1/Robtek/6semester/\
Bachelorproject/BSc-PRO/potato_and_catfood/train/potato/*.jpg'

    potato_fil = glob.glob(path)
    potato_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in potato_fil]

    for img in potato_images:
        _ = find_contours(img)

    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()
