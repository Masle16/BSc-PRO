#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

"""
Module for Back-projection
"""

###### IMPORTS ######
import glob
from pathlib import Path
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

###### GLOBAL VARIABLES ######

###### FUNCTIONS ######
def random_color():
    """ Generate random color """
    rgbl = [255, 0, 0]
    random.shuffle(rgbl)

    return tuple(rgbl)

def show_img(img, window_name, width=352, height=240, wait_key=False):
    """ Show image in certain size """

    resized = cv2.resize(img,
                         (width, height),
                         interpolation=cv2.INTER_NEAREST)

    cv2.imshow(window_name, resized)

    if wait_key is True:
        cv2.waitKey(0)

    return 0

def backproject(hist, img, bgd_mask):
    """
    Performs backprojection on img\n
    Returns region of interest (448 x 448) and bounding rect of found contour\n
    @hist is the normalized histogram of your template converted to hsv\n
    @img is the input image\n
    @bgd_mask is the mask to remove unnessary background\n
    """

    ################# BACK-PROJECTION #################
    _img = cv2.blur(img, (5, 5))
    img_hsv = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([img_hsv], [0, 1], hist, [0, 180, 0, 256], scale=1)

    ################# REMOVE NOISE #################
    _, mask = cv2.threshold(mask, 53, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge((mask, mask, mask))
    result = cv2.bitwise_and(_img, mask)
    result = cv2.bitwise_and(result, bgd_mask)

    ################# FIND CONTOURS #################
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 0, 127, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours is found
    if not contours:
        rows, cols = img.shape[:2]
        x = int(cols / 2)
        y = int(rows / 2)
        width = height = 50
    else:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        index = np.argmax(areas)
        cnt = contours[index]
        x, y, width, height = cv2.boundingRect(cnt)

    ################# FIND REGION 448 x 448 #################
    x_ctr = int((x + (x + width)) / 2)
    y_ctr = int((y + (y + height)) / 2)
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

    # Return region (448 x 448) and bounding rect of found contour
    return (x_left, x_right, y_up, y_down), (x, y, width, height)

###### MAIN ######
def main():
    """ Main function """

    ####### IMPORT IMAGES #######
    path_images = [
        str(Path('dataset3/res_still/test/background/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/potato/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/carrots/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/catfood_salmon/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/catfood_beef/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/bun/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/arm/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/kethchup/*.jpg').resolve())
    ]

    ################## Back-projection ##################
    # Create template histogram
    path = str(Path('preprocessing/template.jpg').resolve())
    template = cv2.imread(path, cv2.IMREAD_COLOR)
    template = cv2.blur(template, (5, 5))
    template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    template_hist = cv2.calcHist(images=[template_hsv],
                                 channels=[0, 1],
                                 mask=None,
                                 histSize=[180, 256],
                                 ranges=[0, 180, 0, 256])
    cv2.normalize(src=template_hist,
                  dst=template_hist,
                  alpha=0,
                  beta=255,
                  norm_type=cv2.NORM_MINMAX)

    # # Plot histogram
    # plt.plot(template_hist)
    # plt.xlim([0, 179])
    # plt.xlabel('Pixel values')
    # plt.ylabel('Intensity')
    # plt.show()

    # channels = ('Hue', 'Saturation', 'Value')
    # for i, channel in enumerate(channels):
    #     if i == 0:
    #         size = [180]
    #         length = [0, 180]
    #     else:
    #         size = [256]
    #         length = [0, 256]

    #     hist = cv2.calcHist([template_hsv], [i], None, size, length)
    #     cv2.normalize(src=hist, dst=hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    #     plt.plot(hist, label=channel)
    #     plt.xlim([0, 256])

    # plt.xlabel('Pixel values')
    # plt.ylabel('Intensity')
    # plt.legend(loc='upper center')
    # plt.show()

    # Import background mask
    path = str(Path('preprocessing/bgd_mask.jpg').resolve())
    mask = cv2.imread(path, cv2.IMREAD_COLOR)

    num = 0
    for path_img in path_images:
        images_fil = glob.glob(path_img)
        images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in images_fil]

        for img in images:
            t1 = cv2.getTickCount()

            roi, coordinates = backproject(hist=template_hist,
                                           img=img,
                                           bgd_mask=mask)

            t2 = cv2.getTickCount()

            clock_cycles = (t2 - t1)
            f = open('preprocessing/output_bp/clock_cycles.txt', 'a')
            txt = str(clock_cycles) + '\n'
            f.write(txt)
            f.close()

            time = clock_cycles / cv2.getTickFrequency()
            f = open('preprocessing/output_bp/time.txt', 'a+')
            txt = str(time) + '\n'
            f.write(txt)
            f.close()

            (x_left, x_right, y_up, y_down) = roi
            (x, y, width, height) = coordinates

            cv2.rectangle(img=img,
                          pt1=(x_left, y_up),
                          pt2=(x_right, y_down),
                          color=(255, 0, 0),
                          thickness=3)

            cv2.rectangle(img=img,
                          pt1=(x, y),
                          pt2=(x + width, y + height),
                          color=(0, 0, 255),
                          thickness=3)

            path = str(Path('preprocessing/output_bp/output_' + str(num) + '.jpg').resolve())
            cv2.imwrite(path, img)

            num += 1

    return 0

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()

