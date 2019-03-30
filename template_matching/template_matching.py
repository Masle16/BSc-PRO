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
POTATO = 0
CARROT = 1
CAT_BEEF = 2
CAT_SAL = 3
BUN = 4
ARM = 5
KETCHUP = 6
BACKGROUND = 7

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

def template_match_meth(template, src):
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

def template_matching(templ, src, mask, method=cv2.TM_SQDIFF):
    """
    Performs template matching with rotation\n
    returns region of found template\n
    @templ the template to search for\n
    @src the source image to search in
    """

    # Store rows and cols for template
    rows, cols = templ.shape[:2]

    # Remove unnessary background
    src = cv2.bitwise_and(src, mask)

    value = None
    for angle in np.arange(0, 360, 45):
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

    bottom_right = (top_left[0] + rows_rotate, top_left[1] + cols_rotate)

    print(value)

    return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

def getRegionOfInterest(category, templ, src, mask):
    """
    returns a region of interest (448 x 448)\n
        @param category is the category to look for (potato, carrot ...)\n
        @param templ is the template to search for\n
        @param src is the source image to search in\n
        @param mask is the mask to remove unnessary background\n
        @param method is the matching metric
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
    elif category == ARM:
        kernel_img = (10, 10)
        kernel_templ = (5, 5)
        method = cv2.TM_SQDIFF
    elif category == KETCHUP:
        kernel_img = (10, 10)
        kernel_templ = (5, 5)
        method = cv2.TM_SQDIFF
    elif category == BUN:
        kernel_img = (10, 10)
        kernel_templ = (5, 5)
        method = cv2.TM_SQDIFF
    else:
        kernel_img = (8, 8)
        kernel_templ = (4, 4)
        method = cv2.TM_SQDIFF

    # Make private copies
    img = src.copy()
    template = templ.copy()

    # Remove unnessary background
    img = cv2.bitwise_and(img, mask)

    ###### CONVERT TO GRAYSCALE, DOWNSCALE AND BLUR ######
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

    cv2.imshow('Input', img)
    cv2.imshow('Template', template)

    ###### TEMPLATE MATCH WITH ROTATION ######
    rows, cols = template.shape[:2]
    value = None

    for angle in np.arange(0, 360, 45):
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
                match_space = res
            elif value > min_val:
                value = min_val
                top_left = min_loc
                match_space = res
        else:
            if value is None:
                value = max_val
                top_left = max_loc
                match_space = res
            elif value < max_val:
                value = max_val
                top_left = max_loc
                match_space = res

    show_img(match_space, 'Matching space')

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

    # Kat laks
    path = str(Path('dataset2/images/catfood_salmon/*.jpg').resolve())
    cat_sal_fil = glob.glob(path)
    cat_sal_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_sal_fil]

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

    # # Ketchup
    # path = str(Path('dataset2/images/kethchup/*.jpg').resolve())
    # ketchup_fil = glob.glob(path)
    # ketchup_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in ketchup_fil]

    # Combine input images and shuffle
    # input_images = carrot_images + potato_images + cat_sal_images + cat_beef_images
    # random.shuffle(input_images)

    ####### IMPORT TEMPLATES #######
    path = str(Path('template_matching/template_potato.jpg').resolve())
    template_potato = cv2.imread(path, cv2.IMREAD_COLOR)

    path = str(Path('template_matching/template_carrot.jpg').resolve())
    template_carrot = cv2.imread(path, cv2.IMREAD_COLOR)

    path = str(Path('template_matching/template_cat_beef.jpg').resolve())
    template_cat_beef = cv2.imread(path, cv2.IMREAD_COLOR)

    path = str(Path('template_matching/template_cat_sal_2.jpg').resolve())
    template_cat_sal = cv2.imread(path, cv2.IMREAD_COLOR)

    ####### IMPORT BACKGROUND MASK #######
    path = str(Path('template_matching/bgd_mask.jpg').resolve())
    bgd_mask = cv2.imread(path, cv2.IMREAD_COLOR)

    ####### TEMPLATE MATCHING #######
    # text = ['Potato', 'Carrot', 'Cat sal']
    # color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for src in cat_sal_images:
        display = src.copy()
        roi = getRegionOfInterest(category=CAT_SAL,
                                  templ=template_cat_sal,
                                  src=src,
                                  mask=bgd_mask)
        (x_left, x_right, y_up, y_down) = roi
        cv2.rectangle(img=display,
                      pt1=(x_left, y_up),
                      pt2=(x_right, y_down),
                      color=(0, 0, 255),
                      thickness=4)
        show_img(display, 'Area', wait_key=True)

    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()
