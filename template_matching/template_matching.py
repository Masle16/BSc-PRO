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
ROI_METHOD = cv2.TM_SQDIFF
TEMPL_METHOD = cv2.TM_SQDIFF

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

def find_roi(templ, src):
    """
    Returns region of interest(448 x 448)\n
    @templ is the template you wish to locate\n
    @src is the source image to locate the template in\n
    @mask is the mask to remove unnessary background
    """

    ####### CONVERT TO GRAYSCALE AND DOWNSCALE #######
    # Template
    template = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
    template = cv2.blur(template, (5, 5))
    height_template, width_template = template.shape[:2]
    dim_template = (int(width_template / DOWNSCALING), int(height_template / DOWNSCALING))
    template = cv2.resize(src=template,
                          dsize=dim_template,
                          dst=None,
                          interpolation=cv2.INTER_CUBIC)

    # Input image
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (5, 5))
    height_img, width_img = img.shape[:2]
    dim_img = (int(width_img / DOWNSCALING), int(height_img / DOWNSCALING))
    img = cv2.resize(src=img,
                     dsize=dim_img,
                     dst=None,
                     interpolation=cv2.INTER_CUBIC)

    ####### CREATE DISTANCE MAP #######
    img = cv2.distanceTransform(src=img,
                                distanceType=cv2.DIST_L2,
                                maskSize=3)
    template = cv2.distanceTransform(src=template,
                                     distanceType=cv2.DIST_L2,
                                     maskSize=3)

    ####### PERFORM TEMPLATE MATCHING #######
    matching_space = cv2.matchTemplate(image=img,
                                       templ=template,
                                       method=ROI_METHOD)

    # Find local minimum and maximum in matching space
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src=matching_space)

    if ROI_METHOD is cv2.TM_SQDIFF or ROI_METHOD is cv2.TM_SQDIFF_NORMED:
        top_left = min_loc
        value = min_val
    else:
        top_left = max_loc
        value = max_val

    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

    ####### UPSCALE #######
    x = top_left[0] * DOWNSCALING
    y = top_left[1] * DOWNSCALING
    width = bottom_right[0] * DOWNSCALING
    height = bottom_right[1] * DOWNSCALING

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

def template_matching(templ, src, mask):
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
                                method=TEMPL_METHOD)

        # Find minimum in matching space
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src=res)

        if TEMPL_METHOD is cv2.TM_SQDIFF or TEMPL_METHOD is cv2.TM_SQDIFF_NORMED:
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

def templ_matching(templ, src):
    """
    Performs template matching\n
    @templ is the template to search for\n
    @src is the source image to search in
    """

    img = src.copy()
    template = templ.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (5, 5))
    height_img, width_img = img.shape[:2]
    img_dim = (int(width_img / DOWNSCALING), int(height_img / DOWNSCALING))
    img = cv2.resize(src=img,
                     dsize=img_dim,
                     interpolation=cv2.INTER_CUBIC)

    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    height_t, width_t = template.shape[:2]
    templ_dim = (int(width_t / DOWNSCALING), int(height_t / DOWNSCALING))
    template = cv2.resize(src=template,
                          dsize=templ_dim,
                          interpolation=cv2.INTER_CUBIC)

    matching_space = cv2.matchTemplate(image=img,
                                       templ=template,
                                       method=TEMPL_METHOD)

    show_img(matching_space, 'Matching space')

    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(src=matching_space)

    if TEMPL_METHOD is cv2.TM_SQDIFF or TEMPL_METHOD is cv2.TM_SQDIFF_NORMED:
        top_left = min_loc
        value = min_val
    else:
        top_left = max_loc
        value = max_val

    (x, y) = top_left
    x *= DOWNSCALING
    y *= DOWNSCALING
    width = templ.shape[1]
    height = templ.shape[0]

    print(value, top_left)

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

    # Kartofler
    path = str(Path('dataset2/images/potato/*.jpg').resolve())
    potato_fil = glob.glob(path)
    potato_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in potato_fil]

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

    # # Ketchup
    # path = str(Path('dataset2/images/kethchup/*.jpg').resolve())
    # ketchup_fil = glob.glob(path)
    # ketchup_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in ketchup_fil]

    # Combine input images and shuffle
    # input_images = carrot_images + potato_images + cat_sal_images + cat_beef_images
    # random.shuffle(input_images)

    ####### IMPORT TEMPLATES #######
    path = str(Path('template_matching/template_potato_2.jpg').resolve())
    template_potato = cv2.imread(path, cv2.IMREAD_COLOR)

    path = str(Path('template_matching/template_carrot.jpg').resolve())
    template_carrot = cv2.imread(path, cv2.IMREAD_COLOR)

    # path = str(Path('template_matching/template_cat_beef.jpg').resolve())
    # template_cat_beef = cv2.imread(path, cv2.IMREAD_COLOR)

    path = str(Path('template_matching/template_cat_sal_2.jpg').resolve())
    template_cat_sal = cv2.imread(path, cv2.IMREAD_COLOR)

    templates = [template_potato, template_carrot, template_cat_sal]

    ####### IMPORT BACKGROUND MASK #######
    path = str(Path('template_matching/bgd_mask.jpg').resolve())
    bgd_mask = cv2.imread(path, cv2.IMREAD_COLOR)

    ####### TEMPLATE MATCHING #######
    for src in potato_images:
        src = cv2.bitwise_and(src, bgd_mask)

        show_img(src, 'Input')

        text = ['Potato', 'Carrot', 'Cat sal']
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for i, temp in enumerate(templates):
            show_img(temp, 'Template')

            display = src.copy()
            cv2.putText(img=display,
                        text=text[i],
                        org=(0, 710),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3,
                        color=color[i],
                        thickness=2,
                        lineType=cv2.LINE_AA)

            roi = templ_matching(templ=temp, src=src)

            (x_left, x_right, y_up, y_down) = roi

            cv2.rectangle(img=display,
                          pt1=(x_left, y_up),
                          pt2=(x_right, y_down),
                          color=color[i],
                          thickness=4)

            show_img(display, 'Area', wait_key=True)

    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()
