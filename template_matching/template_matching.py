#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

""" Module for template matching and chamfer matching """

###### IMPORTS ######
import glob
import random
from pathlib import Path
import cv2
import numpy as np
from background_subtration import background_sub, run_avg

###### GLOBAL VARIABLES ######
DOWNSCALING = 4
CLASSES = ['Potato', 'Carrot', 'Cat beef', 'Cat salmon']
BGD_MASK = cv2.imread(str(Path('preprocessing/background_mask.jpg').resolve()),
                      cv2.IMREAD_GRAYSCALE)
ROI_METHOD = cv2.TM_SQDIFF
TEMPL_METHOD = cv2.TM_SQDIFF_NORMED

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

def thresholding(src):
    """ Threshold image """

    _img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, _img = cv2.threshold(_img, 100, 255, cv2.THRESH_BINARY_INV)

    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    _img = cv2.morphologyEx(_img, cv2.MORPH_OPEN, kernel)
    _img = cv2.bitwise_and(_img, _img, mask=BGD_MASK)
    show_img(_img, 'Image')

    cnts, _ = cv2.findContours(_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    display = src.copy()
    cv2.drawContours(display, cnts, -1, (0, 255, 0), 4)
    show_img(display, 'Contours')

    # Calculate contours pixel intensity
    cnt_pixel_value = []
    for cnt in cnts:
        pixel_sum = 0
        cnt = np.asarray(cnt)
        cnt = cnt.reshape(cnt.shape[0], cnt.shape[2])
        pixel_sum = _img[cnt[:, :][:, 1], cnt[:, :][:, 0]]
        cnt_pixel_value.append(np.sum(pixel_sum))

    # Selected contour with highest pixel intensity
    index = np.argmax(cnt_pixel_value)

    x, y, width, height = cv2.boundingRect(cnts[index])

    return (x, y, width, height)

def find_roi(template, src, method=ROI_METHOD):
    """ returns region of interest(448 x 448) for further inspection """

    # Convert to grayscale and Downscale image and template
    _template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV_FULL)
    height_template, width_template = _template.shape[:2]
    dim_template = (int(width_template / DOWNSCALING), int(height_template / DOWNSCALING))
    _template = cv2.resize(_template,
                           dim_template,
                           interpolation=cv2.INTER_CUBIC)

    _img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV_FULL)
    height_img, width_img = _img.shape[:2]
    dim_img = (int(width_img / DOWNSCALING), int(height_img / DOWNSCALING))
    _img = cv2.resize(_img,
                      dim_img,
                      interpolation=cv2.INTER_CUBIC)

    # # Detect edge in images
    # img_edge = cv2.Canny(img_res, 100, 200)
    # template_edge = cv2.Canny(template_res, 100, 200)

    # # Create distance map
    # img_dist = cv2.distanceTransform(img_edge, cv2.DIST_L2, 3)
    # template_dist = cv2.distanceTransform(template_edge, cv2.DIST_L2, 3)

    matching_space = cv2.matchTemplate(_img,
                                       _template,
                                       method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching_space)

    if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
        top_left = min_loc
        value = min_val
    else:
        top_left = max_loc
        value = max_val

    bottom_right = (top_left[0] + _template.shape[1], top_left[1] + _template.shape[0])
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


def template_matching(template, src, method=TEMPL_METHOD):
    """ Performs template matching with scaling and rotation """

    # # Convert to gray
    # src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # # Rezise template
    # rows, cols = src.shape[:2]
    # dim = (cols, rows)
    # template = cv2.resize(src=template,
    #                       dsize=dim,
    #                       interpolation=cv2.INTER_CUBIC)

    # Store rows and cols for template
    rows, cols = template.shape[:2]

    value = None
    for angle in np.arange(0, 360, 45):
        rotation_matrix = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2),
                                                  angle=angle,
                                                  scale=1)

        template_rotate = cv2.warpAffine(src=template,
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

    bottom_right = (top_left[0] + cols_rotate, top_left[1] + rows_rotate)

    return value

def object_detection(template, roi):
    """ returns the best matching object """

    rows, cols = roi.shape[:2]
    dim = (cols, rows)

    template = cv2.resize(src=template,
                          dsize=dim,
                          interpolation=cv2.INTER_CUBIC)

    if template.shape[:2] != roi.shape[:2]:
        raise ValueError('A very specific bad thing happened.')

    res = cv2.matchTemplate(image=roi,
                            templ=template,
                            method=TEMPL_METHOD)

    # Find minimum in matching space
    min_val, max_val, _, _ = cv2.minMaxLoc(src=res)

    if TEMPL_METHOD is cv2.TM_SQDIFF or TEMPL_METHOD is cv2.TM_SQDIFF_NORMED:
        value = np.absolute(min_val)
    else:
        value = np.absolute(max_val)

    return value

####### MAIN FUNCTION #######
def main():
    """ Main function """

    ####### IMPORT IMAGES #######

    # Baggrund
    path = str(Path('images_1280x720/baggrund/bevægelse/*.jpg').resolve())
    background_fil = glob.glob(path)
    background_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in background_fil]
    bgd_img = run_avg(background_images)

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

    input_images = carrot_images + potato_images + cat_sal_images + cat_beef_images
    random.shuffle(input_images)
    random.shuffle(input_images)
    random.shuffle(input_images)

    ####### IMPORT TEMPLATES #######

    path = str(Path('template_matching/template_potato.jpg').resolve())
    template_potato = cv2.imread(path, cv2.IMREAD_COLOR)

    path = str(Path('template_matching/template_carrot.jpg').resolve())
    template_carrot = cv2.imread(path, cv2.IMREAD_COLOR)

    path = str(Path('template_matching/template_cat_beef.jpg').resolve())
    template_cat_beef = cv2.imread(path, cv2.IMREAD_COLOR)

    # path = str(Path('template_matching/template_cat_sal_2.jpg').resolve())
    # template_cat_sal = cv2.imread(path, cv2.IMREAD_COLOR)

    templates = [template_potato, template_carrot, template_cat_beef]

    # ####### CREATE CHAMFER TEMPLATES #######

    # chamfer_templates = []
    # for temp in templates:

    #     # Check template
    #     if temp is None:
    #         raise Exception('Template equals None')

    #     # Create chamfer template
    #     chamfer = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    #     chamfer = cv2.Canny(chamfer, 100, 200)
    #     _, chamfer = cv2.threshold(chamfer, 127, 255, cv2.THRESH_BINARY_INV)

    #     chamfer = cv2.distanceTransform(chamfer, cv2.DIST_L2, 3)
    #     chamfer_templates.append(chamfer)

    ####### TEMPLATE MATCHING #######

    for src in input_images:
        display_img = src.copy()

        # Find region of interest in image
        roi, cnt = background_sub(img=src,
                                  bgd=bgd_img,
                                  bgd_mask=BGD_MASK)

        (x_left, x_right, y_up, y_down) = roi
        (x, y, width, height) = cnt

        roi = src[y_up : y_down, x_left : x_right]

        cv2.rectangle(img=display_img,
                      pt1=(x_left, y_up),
                      pt2=(x_right, y_down),
                      color=(255, 0, 0),
                      thickness=4)

        cv2.rectangle(img=display_img,
                      pt1=(x, y),
                      pt2=(x + width, y + height),
                      color=(0, 255, 0),
                      thickness=4)

        values = []
        for template in templates:
            value = template_matching(template=template,
                                      src=roi)
            values.append(value)

        print(values)

        if TEMPL_METHOD is cv2.TM_SQDIFF or TEMPL_METHOD is cv2.TM_SQDIFF_NORMED:
            index = np.argmin(values)
        else:
            index = np.argmax(values)

        if index == 0:
            result = 'Potato'
        elif index == 1:
            result = 'Carrot'
        elif index == 2:
            result = 'Cat beef'
        elif index == 3:
            result = 'Cat sal'

        cv2.putText(img=display_img,
                    text=result,
                    org=(0, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA)

        cv2.imshow('Detected object', display_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()
