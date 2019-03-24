#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

""" Module for template matching and chamfer matching """

###### IMPORTS ######
import glob
from pathlib import Path
import cv2
import numpy as np

###### GLOBAL VARIABLES ######
DOWNSCALING = 4
CLASSES = ['Potato', 'Carrot', 'Cat beef', 'Cat salmon']

###### FUNCTIONS ######
def show_img(img, window_name, width=640, height=400, wait_key=False):
    """ Show image in certain size """

    resized = cv2.resize(img,
                         (width, height),
                         interpolation=cv2.INTER_NEAREST)

    cv2.imshow(window_name, resized)

    if wait_key is True:
        cv2.waitKey(0)

    return 0

def template_match_meth(template, src):
    """ displays six methods for template matching and shows how the perform """

    # convert to gray
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    _w, _h = template.shape[: : -1]

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
               'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
               'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = img_gray
        method = eval(meth)

        # Apply template matching
        res = cv2.matchTemplate(img, template, method)
        _, _, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + _w, top_left[1] + _h)

        print(method)

        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        cv2.imshow('Matching Result', res)
        cv2.imshow('Detected Point', img)
        cv2.waitKey(0)

    return img

def find_roi(template, src, method=cv2.TM_CCORR_NORMED):
    """ returns region of interest(448 x 448) for further inspection """

    # Convert to grayscale and Downscale image and template
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    height_template, width_template = template.shape[:2]
    dim_template = (int(width_template / DOWNSCALING), int(height_template / DOWNSCALING))
    template_res = cv2.resize(template_gray,
                              dim_template,
                              interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    height_img, width_img = src.shape[:2]
    dim_img = (int(width_img / DOWNSCALING), int(height_img / DOWNSCALING))
    img_res = cv2.resize(img_gray,
                         dim_img,
                         interpolation=cv2.INTER_CUBIC)

    # Detect edge in images
    img_edge = cv2.Canny(img_res, 100, 200)
    template_edge = cv2.Canny(template_res, 100, 200)

    # Create distance map
    img_dist = cv2.distanceTransform(img_edge, cv2.DIST_L2, 3)
    template_dist = cv2.distanceTransform(template_edge, cv2.DIST_L2, 3)

    matching_space = cv2.matchTemplate(img_dist,
                                       template_dist,
                                       method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching_space)

    top_left = max_loc

    bottom_right = (top_left[0] + template_res.shape[1], top_left[1] + template_res.shape[0])
    top_left_x = top_left[0] * DOWNSCALING
    top_left_y = top_left[1] * DOWNSCALING
    bottom_right_x = bottom_right[0] * DOWNSCALING
    bottom_right_y = bottom_right[1] * DOWNSCALING

    # Crop region of interest
    radius = 224
    x_ctr = int((top_left_x + bottom_right_x) / 2)
    y_ctr = int((top_left_y + bottom_right_y) / 2)
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

    return (x_left, x_right, y_up, y_down)


def template_matching(template, src, method=cv2.TM_SQDIFF):
    """ Performs template matching with scaling and rotation """

    rows, cols = template.shape[:2]
    value = None
    for angle in np.arange(0, 360, 45):
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                                  angle,
                                                  1)
        template_rotate = cv2.warpAffine(template,
                                         rotation_matrix,
                                         (cols, rows))
        matching_space = cv2.matchTemplate(src,
                                           template_rotate,
                                           method)

        # Find minimum in matching space
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching_space)

        if value is None:
            value = min_val
            top_left = min_loc
            rows_rotate, cols_rotate = template_rotate.shape[:2]
        elif value > min_val:
            value = min_val
            top_left = min_loc
            rows_rotate, cols_rotate = template_rotate.shape[:2]

    bottom_right = (top_left[0] + cols_rotate, top_left[1] + rows_rotate)

    return (top_left, bottom_right)

####### MAIN FUNCTION #######
def main():
    """ Main function """

    ####### IMPORT IMAGES #######

    # Baggrund
    path = str(Path('images_1280x720/baggrund/bevægelse/*.jpg').resolve())
    background_fil = glob.glob(path)
    background_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in background_fil]

    # Guleroedder
    path = str(Path('images_1280x720/gulerod/still/*.jpg').resolve())
    carrot_fil = glob.glob(path)
    carrot_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in carrot_fil]

    # Kartofler
    path = str(Path('images_1280x720/kartofler/still/*.jpg').resolve())
    potato_fil = glob.glob(path)
    potato_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in potato_fil]

    # Kat laks
    path = str(Path('images_1280x720/kat_laks/still/*.jpg').resolve())
    cat_sal_fil = glob.glob(path)
    cat_sal_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_sal_fil]

    # Kat okse
    path = str(Path('images_1280x720/kat_okse/still/*.jpg').resolve())
    cat_beef_fil = glob.glob(path)
    cat_beef_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_beef_fil]

    ####### IMPORT TEMPLATES #######

    path = str(Path('template_matching/template_potato.jpg').resolve())
    template_potato = cv2.imread(path, cv2.IMREAD_COLOR)

    path = str(Path('template_matching/template_carrot.jpg').resolve())
    template_carrot = cv2.imread(path, cv2.IMREAD_COLOR)

    path = str(Path('template_matching/template_cat_beef.jpg').resolve())
    template_cat_beef = cv2.imread(path, cv2.IMREAD_COLOR)

    path = str(Path('template_matching/template_cat_sal.jpg').resolve())
    template_cat_sal = cv2.imread(path, cv2.IMREAD_COLOR)

    templates = [template_potato, template_carrot, template_cat_beef, template_cat_sal]

    ####### CREATE CHAMFER TEMPLATES #######

    chamfer_templates = []
    for temp in templates:

        # Check template
        if temp is None:
            raise Exception('Template equals None')

        # Create chamfer template
        chamfer = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        chamfer = cv2.Canny(chamfer, 100, 200)
        _, chamfer = cv2.threshold(chamfer, 127, 255, cv2.THRESH_BINARY_INV)

        chamfer = cv2.distanceTransform(chamfer, cv2.DIST_L2, 3)
        chamfer_templates.append(chamfer)

    ####### IMPORT BACKGROUND MASK #######

    path = str(Path('preprocessing/background_mask.jpg').resolve())
    background_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    ####### TEMPLATE MATCHING #######

    for img in potato_images:


        for template in templates:

            display_img = img.copy()

            (x_left, x_right, y_up, y_down) = find_roi(template, img)

            cv2.rectangle(display_img,
                          (x_left, y_up),
                          (x_right, y_down),
                          (255, 0, 0),
                          4)

            roi = img[y_up : y_down, x_left : x_right]
            (top_left, bottom_right) = template_matching(template, roi)

            cv2.rectangle(display_img,
                          (top_left[0] + x_left, top_left[1] + y_up),
                          (bottom_right[0] + x_left, bottom_right[1] + y_up),
                          (0, 0, 255),
                          4)

            cv2.imshow('Detected area', display_img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
