#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

"""
Module for template matching and chamfer matching
"""

###### IMPORTS ######
import glob
import random
from pathlib import Path
import cv2
import numpy as np
import time

###### GLOBAL VARIABLES ######
DOWNSCALING = 4
CLASSES = ['Potato', 'Carrot', 'Cat beef', 'Cat salmon']
# BACKGROUND = 0
POTATO = 0
CARROT = 1
CAT_SAL = 2
CAT_BEEF = 3
BUN = 4
ARM = 5
KETCHUP = 6

###### FUNCTIONS ######
def show_img(img, window_name, width=352, height=240, wait_key=False):
    """
    Show image in certain size
    """

    resized = cv2.resize(src=img,
                         dsize=(width, height),
                         dst=None,
                         interpolation=cv2.INTER_NEAREST)

    cv2.imshow(window_name, resized)

    if wait_key is True:
        cv2.waitKey(0)

    return 0

def compareMatchingMetrics(template, src):
    """
    displays six methods for template matching and shows how they perform
    """

    # convert to gray
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    width, height = template.shape[: : -1]

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
               'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
               'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = img_gray
        method = eval(meth)

        # Apply template matching
        res = cv2.matchTemplate(image=img,
                                templ=template,
                                method=method)

        _, _, min_loc, max_loc = cv2.minMaxLoc(src=res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + width, top_left[1] + height)

        print(method)

        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        cv2.imshow('Matching Result', res)
        cv2.imshow('Detected Point', img)
        cv2.waitKey(0)

    return img

def removeBackground(src, bgd_mask):
    """
    returns image without unnessary background\n
        @param src is the input image\n
        @param bgd_mask is the mask to remove the top corners
    """

    result = src.copy()
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lower = (0, 70, 0)
    upper = (179, 255, 255)
    mask = cv2.inRange(src=hsv, lowerb=lower, upperb=upper)
    result = cv2.bitwise_and(result, result, mask=mask)
    result = cv2.bitwise_and(result, bgd_mask)

    return result

def findTemplate(templ, src):
    """
    returns region of found template\n
        @param category is the category to search for\n
        @param templ the template to search for\n
        @param src the source image to search in
    """

    kernel_img = (10, 10)
    kernel_templ = (5, 5)
    method = cv2.TM_CCORR_NORMED

    # Store rows and cols for template
    rows, cols = templ.shape[:2]

    # Blur image
    img = cv2.blur(src=src, ksize=kernel_img)
    template = cv2.blur(src=templ, ksize=kernel_templ)

    value = None
    for angle in np.linspace(start=0, stop=360, num=24):
        rotation_matrix = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2),
                                                  angle=angle,
                                                  scale=1)

        template_rotate = cv2.warpAffine(src=template,
                                         M=rotation_matrix,
                                         dsize=(cols, rows))

        res = cv2.matchTemplate(image=img,
                                templ=template_rotate,
                                method=method)

        # Find minimum in matching space
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src=res)

        if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
            if value is None:
                value = min_val
                top_left = min_loc
                rows_rotate, cols_rotate = template_rotate.shape[:2]
                match_space = res.copy()
            elif value > min_val:
                value = min_val
                top_left = min_loc
                rows_rotate, cols_rotate = template_rotate.shape[:2]
                match_space = res.copy()
        else:
            if value is None:
                value = max_val
                top_left = max_loc
                rows_rotate, cols_rotate = template_rotate.shape[:2]
                match_space = res.copy()
            elif value < max_val:
                value = max_val
                top_left = max_loc
                rows_rotate, cols_rotate = template_rotate.shape[:2]
                match_space = res.copy()

    # # Display matching space
    # print('Matching space size:', match_space.shape)
    # cv2.imshow('Matching space', cv2.normalize(match_space,
    #                                            None,
    #                                            alpha=0,
    #                                            beta=1,
    #                                            norm_type=cv2.NORM_MINMAX))
    # cv2.waitKey()

    (x, y) = top_left
    width = cols_rotate
    height = rows_rotate

    return value, (x, y, width, height)

def getRegionOfInterest(templ, src):
    """
    returns a region of interest (448 x 448)\n
        @param category is the category to look for (potato, carrot ...)\n
        @param templ is the template to search for\n
        @param src is the source image to search in\n
    """

    kernel_img = (10, 10)
    kernel_templ = (5, 5)
    method = cv2.TM_CCORR_NORMED

    # Make private copies
    img = src.copy()
    template = templ.copy()

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Downscale
    height_img, width_img = img.shape[:2]
    img_dim = (int(width_img / DOWNSCALING), int(height_img / DOWNSCALING))
    img = cv2.resize(src=img,
                     dsize=img_dim,
                     interpolation=cv2.INTER_CUBIC)

    height_t, width_t = template.shape[:2]
    templ_dim = (int(width_t / DOWNSCALING), int(height_t / DOWNSCALING))
    template = cv2.resize(src=template,
                          dsize=templ_dim,
                          interpolation=cv2.INTER_CUBIC)

    # print('Image size:', img.shape)
    # print('Template size:', template.shape)

    # Blur image
    img = cv2.blur(src=img, ksize=kernel_img)
    template = cv2.blur(src=template, ksize=kernel_templ)

    ###### TEMPLATE MATCHING WITH ROTATION ######
    rows, cols = template.shape[:2]
    value = None

    for angle in np.linspace(start=0, stop=360, num=10):
        rotation_matrix = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2),
                                                  angle=angle,
                                                  scale=1)

        template = cv2.warpAffine(src=template,
                                  M=rotation_matrix,
                                  dsize=(cols, rows))

        res = cv2.matchTemplate(image=img,
                                templ=template,
                                method=method)

        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(src=res)

        if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
            if value is None:
                value = min_val
                top_left = min_loc
                match_space = res.copy()
            elif value > min_val:
                value = min_val
                top_left = min_loc
                match_space = res.copy()
        else:
            if value is None:
                value = max_val
                top_left = max_loc
                match_space = res.copy()
            elif value < max_val:
                value = max_val
                top_left = max_loc
                match_space = res.copy()

    # print('Matching space size:', match_space.shape)
    # cv2.imshow('Matching space', cv2.normalize(match_space,
    #                                            None,
    #                                            alpha=0,
    #                                            beta=1,
    #                                            norm_type=cv2.NORM_MINMAX))
    # cv2.waitKey()

    (x, y) = top_left
    x *= DOWNSCALING
    y *= DOWNSCALING
    width = templ.shape[1]
    height = templ.shape[0]

    ####### FIND REGION OF INTEREST (448 x 448) #######
    x_ctr = int((x + (x + width)) / 2)
    y_ctr = int((y + (y + height)) / 2)

    radius = 224
    x_left = x_ctr - radius
    x_right = x_ctr + radius
    y_up = y_ctr - radius
    y_down = y_ctr + radius

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

    # Return region of interest (448 x 448)
    return (x_left, x_right, y_up, y_down)

####### MAIN FUNCTION #######
def main():
    """
    Main function\n
    Shows how to use template matching
    """

    ####### IMPORT IMAGES #######
    path_images = [
        # str(Path('dataset3/res_still/test/background/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/potato/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/carrots/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/catfood_salmon/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/catfood_beef/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/bun/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/arm/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/ketchup/*.jpg').resolve())
    ]

    path_templates = [
        # str(Path('template_matching/templates/template_background.jpg').resolve()),
        str(Path('template_matching/templates/template_potato.jpg').resolve()),
        str(Path('template_matching/templates/template_carrot.jpg').resolve()),
        str(Path('template_matching/templates/template_cat_sal.jpg').resolve()),
        str(Path('template_matching/templates/template_cat_beef.jpg').resolve()),
        str(Path('template_matching/templates/template_bun.jpg').resolve()),
        str(Path('template_matching/templates/template_arm.jpg').resolve()),
        str(Path('template_matching/templates/template_ketchup.jpg').resolve())
    ]

    ####### IMPORT BACKGROUND MASK #######
    path = str(Path('template_matching/bgd_mask.jpg').resolve())
    bgd_mask = cv2.imread(path, cv2.IMREAD_COLOR)

    ####### TEMPLATE MATCHING #######
    text = [
        # 'background',
        'Potato',
        'Carrot',
        'Cat sal',
        'Cat beef',
        'Bun',
        'Arm',
        'Ketchup'
    ]

    cnts = []
    rois = []
    values = []
    # correct = 0
    # times = []

    # for i, path_img in enumerate(path_images):
    #     images_fil = glob.glob(path_img)
    #     images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in images_fil]

    # for j, src in enumerate(images):
    #     t1 = time.time()

    img = cv2.imread('/home/mathi/Desktop/input_image.jpg', cv2.IMREAD_COLOR)
    output = img.copy()

    for k, path_temp in enumerate(path_templates):
        template = cv2.imread(path_temp, cv2.IMREAD_COLOR)

        # Remove unnessary background
        img = removeBackground(img, bgd_mask)
        # img = cv2.bitwise_and(img, bgd_mask)

        # Get region of interest
        roi = getRegionOfInterest(templ=template,
                                  src=img)
        rois.append(roi)

        # rois.append(roi)
        (x_left, x_right, y_up, y_down) = roi
        roi = img[y_up : y_down, x_left : x_right]

        # Find template in region
        value, cnt = findTemplate(templ=template,
                                  src=roi)

        values.append(value)
        cnts.append(cnt)

    index = np.argmax(values)

    (x_left, x_right, y_up, y_down) = rois[index]
    (x, y, w, h) = cnts[index]
    cv2.rectangle(output, (x_left + x, y_up + y), (x_left + x + w, y_up + y + h), (0, 0, 255), 8)

    score = str(round(values[index], 3))
    txt = text[index] + ': ' + score
    cv2.putText(output, txt, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0,255), 4, cv2.LINE_AA)

    cv2.imshow('Output', output)
    cv2.waitKey()

    # t2 = time.time()
    # times.append(t2 - t1)

    # if index == i:
    #     correct += 1

    # print(
    #     'Status:', text[i],
    #     ', number:', j, '/', 431,
    #     ', correct:', correct,
    #     ', time used:', (t2 - t1), 'seconds'
    # )

    # values.clear()

    # print('Number of correct:', correct, '/', 431)
    # print('Accuracy:', (correct / 431))
    # print('Average time:', np.mean(times))

    return 0

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
