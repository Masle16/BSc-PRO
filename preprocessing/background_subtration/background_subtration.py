#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

"""
Module for background subtration
"""

import glob
import cv2
import numpy as np
#from matplotlib import pyplot as plt

def show_img(img, window_name, width=640, height=480, wait_key=False):
    """ Show image in certain size """

    resized = cv2.resize(img,
                         (width, height),
                         interpolation=cv2.INTER_NEAREST)

    cv2.imshow(window_name, resized)

    if wait_key is True:
        cv2.waitKey(0)

    return 0

def remove_background(img):
    """ returns image with no background, only table """

    # Find background pixels coordinates
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 64), (179, 51, 255))
    result = cv2.bitwise_and(img, img, mask=mask)

    return mask, result

def run_avg(path):
    """
    returns running average of all images in path folder
    @path, a string of the path to the folder with images (jpg)
    """

    path_img = path + '/*jpg'
    files = glob.glob(path_img)
    file_images = [cv2.imread(img) for img in files]

    avg = np.float32(file_images[0])

    for img in file_images:
        cv2.accumulateWeighted(img, avg, 0.1)

    result = cv2.convertScaleAbs(avg)

    return result

def background_sub(img, background, background_mask):
    """
    returns cropped image(448 x 448) of region of interest
    @img, the image of interest
    """

    # Remove unessesary background
    _img = cv2.bitwise_and(img, img, mask=background_mask)

    # Calculate image difference and find largest contour
    diff = cv2.absdiff(background, _img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(diff_gray, 50, 255, 0)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # #################
    # # DRAW CONTOURS #
    # #################
    # cnt_img = img.copy()
    # cv2.drawContours(cnt_img, cnts, -1, (0, 255, 0), 3)
    # show_img(cnt_img, 'Contours')

    areas = [cv2.contourArea(cnt) for cnt in cnts]
    max_idx = np.argmax(areas)
    cnt = cnts[max_idx]

    _x, _y, _w, _h = cv2.boundingRect(cnt)
    x_ctr = int((_x + (_x + _w)) / 2)
    y_ctr = int((_y + (_y + _h)) / 2)
    radius = 224
    x_left = x_ctr - radius
    x_right = x_ctr + radius
    y_up = y_ctr - radius
    y_down = y_ctr + radius

    if x_right > img.shape[1]:
        margin = -1 * (img.shape[1] - x_right)
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
    elif y_down > img.shape[0]:
        margin = -1 * (img.shape[0] - y_down)
        y_down -= margin
        y_up -= margin

    # Display detected area
    img_rect = img.copy()
    cv2.rectangle(img_rect, (x_left, y_up), (x_right, y_down), (0, 0, 255), 4)
    show_img(img_rect, 'Detected area', wait_key=True)

    # Get region of interest
    roi = img[y_up : y_down, x_left : x_right]

    return roi

def main():
    """ main function """

    # Baggrund
    background_fil = glob.glob('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/images_1280x720/baggrund/bevægelse/*.jpg')
    background_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in background_fil]

    # Guleroedder
    carrot_fil = glob.glob('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/images_1280x720/gulerod/still/*.jpg')
    carrot_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in carrot_fil]

    # Kartofler
    potato_fil = glob.glob('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/potato_and_catfood/train/potato/*.jpg')
    potato_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in potato_fil]

    # Kat laks
    cat_sal_fil = glob.glob('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/images_1280x720/kat_laks/still/*.jpg')
    cat_sal_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_sal_fil]

    # Kat okse
    cat_beef_fil = glob.glob('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/images_1280x720/kat_okse/still/*.jpg')
    cat_beef_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_beef_fil]

    background_img = run_avg('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/images_1280x720/baggrund/bevægelse')
    background_mask, background_img = remove_background(background_img)

    _d = 0
    for img in cat_beef_images:
        roi = background_sub(img, background_img, background_mask)

        # path = '/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/background_models/carrot/carrot_%d.jpg' %_d
        # cv2.imwrite(path, roi)

        # _d += 1

    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()
