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

def findTemplate(category, templ, src):
    """
    returns region of found template\n
        @param category is the category to search for\n
        @param templ the template to search for\n
        @param src the source image to search in
    """

    # # Customized setting
    # if category == POTATO:
    #     method = cv2.TM_SQDIFF
    # elif category == CARROT:
    #     method = cv2.TM_SQDIFF
    # elif category == CAT_SAL:
    #     method = cv2.TM_CCOEFF
    # elif category == CAT_BEEF:
    #     method = cv2.TM_CCOEFF
    # # elif category == ARM:
    # #     method = cv2.TM_CCORR
    # elif category == KETCHUP:
    #     method = cv2.TM_CCOEFF
    # elif category == BUN:
    #     method = cv2.TM_SQDIFF
    # # elif category == BACKGROUND:
    # #     method = cv2.TM_SQDIFF
    # else:
    #     method = cv2.TM_SQDIFF

    method = cv2.TM_CCORR_NORMED

    # Store rows and cols for template
    rows, cols = templ.shape[:2]

    value = None
    for angle in np.linspace(start=0, stop=360, num=24):
        rotation_matrix = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2),
                                                  angle=angle,
                                                  scale=1)

        template_rotate = cv2.warpAffine(src=templ,
                                         M=rotation_matrix,
                                         dsize=(cols, rows))

        res = cv2.matchTemplate(image=src,
                                templ=template_rotate,
                                method=method)

        # Find minimum in matching space
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src=res)

        if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
            if value is None:
                value = min_val
                top_left = min_loc
                rows_rotate, cols_rotate = template_rotate.shape[:2]
            elif value > min_val:
                value = min_val
                top_left = min_loc
                rows_rotate, cols_rotate = template_rotate.shape[:2]
        else:
            if value is None:
                value = max_val
                top_left = max_loc
                rows_rotate, cols_rotate = template_rotate.shape[:2]
            elif value < max_val:
                value = max_val
                top_left = max_loc
                rows_rotate, cols_rotate = template_rotate.shape[:2]

    (x, y) = top_left
    width = cols_rotate
    height = rows_rotate

    return value, (x, y, width, height)

def getRegionOfInterest(category, templ, src):
    """
    returns a region of interest (448 x 448)\n
        @param category is the category to look for (potato, carrot ...)\n
        @param templ is the template to search for\n
        @param src is the source image to search in\n
    """

    # Customized setting
    if category == POTATO:
        kernel_img = (10, 10)
        kernel_templ = (5, 5)
        method = cv2.TM_SQDIFF
    elif category == CARROT:
        kernel_img = (10, 10)
        kernel_templ = (5, 5)
        method = cv2.TM_SQDIFF
    elif category == CAT_SAL:
        kernel_img = (10, 10)
        kernel_templ = (5, 5)
        method = cv2.TM_CCORR_NORMED
    elif category == CAT_BEEF:
        kernel_img = (10, 10)
        kernel_templ = (5, 5)
        method = cv2.TM_CCORR_NORMED
    # elif category == ARM:
    #     kernel_img = (10, 10)
    #     kernel_templ = (5, 5)
    #     method = cv2.TM_CCOEFF_NORMED
    elif category == KETCHUP:
        kernel_img = (10, 10)
        kernel_templ = (5, 5)
        method = cv2.TM_CCORR_NORMED
    elif category == BUN:
        kernel_img = (10, 10)
        kernel_templ = (5, 5)
        method = cv2.TM_SQDIFF
    # elif category == BACKGROUND:
    #     kernel_img = (10, 10)
    #     kernel_templ = (5, 5)
    #     method = cv2.TM_SQDIFF
    else:
        kernel_img = (8, 8)
        kernel_templ = (4, 4)
        method = cv2.TM_SQDIFF

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
            elif value > min_val:
                value = min_val
                top_left = min_loc
        else:
            if value is None:
                value = max_val
                top_left = max_loc
            elif value < max_val:
                value = max_val
                top_left = max_loc

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

def calculateDiff(templ, cnt):
    """
    return pixel intensity in calculated difference image
        @param template is the template
        @param src is the found contour
    """

    img = cnt.copy()
    template = templ.copy()

    result = cv2.absdiff(src1=img, src2=template)
    result = np.sum(cv2.sumElems(result)) / result.size

    return result

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

    num = 0
    for path_img in path_images:

        images_fil = glob.glob(path_img)
        images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in images_fil]

        for src in images:
            t1 = cv2.getTickCount()

            for i, path_temp in enumerate(path_templates):
                template = cv2.imread(path_temp, cv2.IMREAD_COLOR)

                # # Remove unnessary background
                img = removeBackground(src, bgd_mask)

                # Get region of interest
                roi = getRegionOfInterest(category=i,
                                          templ=template,
                                          src=img)
                rois.append(roi)
                (x_left, x_right, y_up, y_down) = roi
                roi = img[y_up : y_down, x_left : x_right]

                # Find template in region
                value, cnt = findTemplate(category=i,
                                          templ=template,
                                          src=roi)
                cnts.append(cnt)
                (x, y, width, height) = cnt
                cnt = src[y : y + height, x : x + width]

                values.append(value)

            index = np.argmax(values)

            t2 = cv2.getTickCount()

            clock_cycles = (t2 - t1)
            f = open('template_matching/output_tm/clock_cycles.txt', 'a')
            txt = str(clock_cycles) + '\n'
            f.write(txt)
            f.close()

            time = clock_cycles / cv2.getTickFrequency()
            f = open('template_matching/output_tm/time.txt', 'a+')
            txt = str(time) + '\n'
            f.write(txt)
            f.close()

            (x_left, x_right, y_up, y_down) = rois[index]
            (x, y, width, height) = cnts[index]
            cv2.rectangle(img=src,
                          pt1=(x + x_left, y + y_up),
                          pt2=(x + width + x_left, y + height + y_up),
                          color=(0, 0, 255),
                          thickness=3)
            cv2.putText(img=src,
                        text=text[index],
                        org=(0, 700),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=(255, 0, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA)

            path = str(Path('template_matching/output_tm/output_' + str(num) + '.jpg').resolve())
            cv2.imwrite(path, src)

            print(num)

            rois.clear()
            cnts.clear()
            values.clear()

            num += 1

    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()
