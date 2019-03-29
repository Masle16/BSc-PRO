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
# from matplotlib import pyplot as plt

###### GLOBAL VARIABLES ######
NUMBER = 0

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
    show_img(mask, 'Unfiltered', wait_key=True)

    ################# REMOVE NOISE #################
    _, mask = cv2.threshold(mask, 53, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge((mask, mask, mask))
    result = cv2.bitwise_and(_img, mask)
    result = cv2.bitwise_and(result, bgd_mask)
    show_img(result, 'Result', wait_key=True)

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

    ################## IMPORT IMAGES ##################
    # # Baggrund
    # path = str(Path('dataset2/images/baggrund/*.jpg').resolve())
    # background_fil = glob.glob(path)
    # background_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in background_fil]

    # # Guleroedder
    # path = str(Path('dataset2/images/carrots/*.jpg'))
    # carrot_fil = glob.glob(path)
    # carrot_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in carrot_fil]

    # # Kartofler
    # path = str(Path('dataset2/images/potato/*.jpg').resolve())
    # potato_fil = glob.glob(path)
    # potato_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in potato_fil]

    # # Kat laks
    # path = str(Path('dataset2/images/catfood_salmon/*.jpg').resolve())
    # cat_sal_fil = glob.glob(path)
    # cat_sal_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_sal_fil]

    # # Kat okse
    # path = str(Path('dataset2/images/catfood_beef/*.jpg').resolve())
    # cat_beef_fil = glob.glob(path)
    # cat_beef_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_beef_fil]

    # # Boller
    # path = str(Path('dataset2/images/bun/*.jpg').resolve())
    # bun_fil = glob.glob(path)
    # bun_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in bun_fil]

    # # Arm
    # path = str(Path('dataset2/images/arm/*.jpg').resolve())
    # arm_fil = glob.glob(path)
    # arm_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in arm_fil]

    # Ketchup
    path = str(Path('dataset2/images/kethchup/*.jpg').resolve())
    ketchup_fil = glob.glob(path)
    ketchup_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in ketchup_fil]

    # # Combine images
    # input_images = (background_images +
    #                 carrot_images +
    #                 potato_images +
    #                 cat_sal_images +
    #                 cat_beef_images +
    #                 bun_images +
    #                 arm_images +
    #                 ketchup_images)

    # # Shuffle
    # random.shuffle(input_images)

    ################## Back-projection ##################
    # Create template histogram (Use template_all.jpg)
    path = str(Path('preprocessing/template.jpg').resolve())
    template = cv2.imread(path, cv2.IMREAD_COLOR)
    template = cv2.blur(template, (5, 5))
    show_img(template, 'Template')
    template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    template_hist = cv2.calcHist([template_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    template_hist = cv2.normalize(template_hist, template_hist, 0, 255, cv2.NORM_MINMAX)

    # Import background mask
    path = str(Path('preprocessing/bgd_mask.jpg').resolve())
    mask = cv2.imread(path, cv2.IMREAD_COLOR)

    for img in ketchup_images:
        show_img(img, 'Input')

        roi, coordinates = backproject(hist=template_hist,
                                       img=img,
                                       bgd_mask=mask)

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

        show_img(img, 'Detected area', wait_key=True)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
