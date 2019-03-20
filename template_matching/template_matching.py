#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

"""
Module for template matching and chamfer matching
"""

###### IMPORTS ######
import glob
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
import cv2
import imutils
import numpy as np

###### GLOBAL VARIABLES ######
CLASSES = ['Potato', 'Carrot', 'Cat beef', 'Cat salmon']

def write_text(img, txt, point=(0, 0)):
    """ Functions for writing text in image """

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("arial.ttf", 50)
    draw.text(point, txt, (255, 0, 0), font=font)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return result

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
    """
    displays six methods for template matching and shows how the perform
    @template, the object you are searching for
    @src, the source image you are searching in
    """

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

def template_matching(templates, src, background_mask, method=cv2.TM_CCORR_NORMED):
    """
    returns crop image of interest
    @template, the template you are searching for
    @src, the source image you are searching in
    @method, the method you are using (preset to cv2.TM_SQDIFF)
        - cv2.TM_CCOEFF
        - cv2.TM_CCOEFF_NORMED
        - cv2.TM_CCORR
        - cv2.TM_CCORR_NORMED
        - cv2.TM_SQDIFF
        - cv2.TM_SQDIFF_NORMED)

    Uses multiscaling to scale image
    """

    img_class = None
    for i, template in enumerate(templates):
        # Store height and width
        height = template.shape[0]
        width = template.shape[1]

        # Resize template
        ratio = 224.0 / template.shape[1]
        dim = (224, int(template.shape[0] * ratio))
        template = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)

        found = None
        for scale in np.linspace(1.0, 2.0, 5)[::-1]:

            # Resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = cv2.resize(template,
                                 (int(dim[0] * scale), int(dim[1] * scale)),
                                 interpolation=cv2.INTER_AREA)

            # If the resized image is smaller than the template, then break from the loop
            if resized.shape[0] > src.shape[0] or resized.shape[1] > src.shape[1]:
                continue

            # Rotate image to different angles
            for angle in np.arange(0, 360, 45):
                rotated = imutils.rotate_bound(resized, angle)

                # Detect edges in the resized grayscale image and apply template
                # matching to find the template in the image
                matching_space = cv2.matchTemplate(src, rotated, method)

                # Find local minima
                matching_space = cv2.normalize(matching_space, 0, 255, cv2.NORM_MINMAX)
                matching_space = cv2.GaussianBlur(matching_space, (55, 55), 0)
                (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(matching_space)

                # If we have found a new maximum correlation value, then update
                # the bookkeeping variable
                if found is None:
                    found = (min_val, max_val, min_loc, max_loc, resized.shape[0], resized.shape[1])
                    img_class = CLASSES[i]

                if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
                    if min_val < found[0]:
                        found = (min_val, max_val, min_loc, max_loc, resized.shape[0], resized.shape[1])
                        img_class = CLASSES[i]
                else:
                    if max_val > found[1]:
                        found = (min_val, max_val, min_loc, max_loc, resized.shape[0], resized.shape[1])
                        img_class = CLASSES[i]

    # Unpack the found variable and compute the (x,y) coordinates of the bounding
    # rect based of the resized ratio

    (min_val, max_val, min_loc, max_loc, height, width) = found

    if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + width, top_left[1] + height)

    # Crop region of interest
    radius = 224
    x_ctr = int((top_left[0] + bottom_right[0]) / 2)
    y_ctr = int((top_left[1] + bottom_right[1]) / 2)
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

    # # Display detected area
    # img_rect = src.copy()
    # cv2.rectangle(img_rect, (x_left, y_up), (x_right, y_down), (0, 0, 255), 4)
    # cv2.imshow('Bounding rect', img_rect)
    # cv2.waitKey(0)
    # #cv2.imwrite('/home/mathi/Desktop/img_rect.jpg', img_rect)

    # img_crop = src[y_up : y_down, x_left : x_right]

    return top_left, bottom_right, img_class

def chamfer_matching(templates, src, background_mask, method=cv2.TM_SQDIFF):
    """
    returns crop image of interest
    @template, the distance map of the template you are searching for
    @src, the source image you are searching in
    @method, the method you are using (preset to cv2.TM_SQDIFF)
        - cv2.TM_CCOEFF
        - cv2.TM_CCOEFF_NORMED
        - cv2.TM_CCORR
        - cv2.TM_CCORR_NORMED
        - cv2.TM_SQDIFF
        - cv2.TM_SQDIFF_NORMED)
    """

    # Remove unnessesary background
    img = cv2.bitwise_and(src, src, mask=background_mask)

    # Convert to src to distance map
    _chamfer_src = cv2.Canny(img, 100, 200)
    _, _chamfer_src = cv2.threshold(_chamfer_src, 127, 255, cv2.THRESH_BINARY_INV)
    img_dist = cv2.distanceTransform(_chamfer_src, cv2.DIST_L2, 3)

    img_class = None
    found = None
    for i, template in enumerate(templates):
        # Store width and height of template
        width = template.shape[1]
        height = template.shape[0]

        # Apply template matching
        matching_space = cv2.matchTemplate(img_dist, template, method)

        # Find local maximum
        matching_space = cv2.normalize(matching_space, 0, 255, cv2.NORM_MINMAX)
        matching_space = cv2.GaussianBlur(matching_space, (55, 55), 0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching_space)

        # If we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None:
            found = (min_val, max_val, min_loc, max_loc, height, width)
            img_class = CLASSES[i]

        if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
            if min_val < found[0]:
                found = (min_val, max_val, min_loc, max_loc, height, width)
                img_class = CLASSES[i]
        else:
            if max_val > found[1]:
                found = (min_val, max_val, min_loc, max_loc, height, width)
                img_class = CLASSES[i]

    # Unpack the found variable and compute the (x,y) coordinates of the bounding
    # rect based of the resized ratio

    (min_val, max_val, min_loc, max_loc, height, width) = found

    if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + width, top_left[1] + height)

    # # Draw rect
    # img_rect = write_text(src, img_class)
    # cv2.rectangle(img_rect, top_left, bottom_right, (0, 0, 255), 4)
    # show_img(img_rect, 'Detected area', wait_key=True)

    # Crop region of interest
    radius = 224
    x_ctr = int((top_left[0] + bottom_right[0]) / 2)
    y_ctr = int((top_left[1] + bottom_right[1]) / 2)
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

    img_crop = src[y_up : y_down, x_left : x_right]

    return img_crop, (x_left, y_up), (x_right, y_down)

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
        chamfer = cv2.Canny(temp, 100, 200)
        _, chamfer = cv2.threshold(chamfer, 127, 255, cv2.THRESH_BINARY_INV)

        chamfer = cv2.distanceTransform(chamfer, cv2.DIST_L2, 3)
        chamfer_templates.append(chamfer)

    ####### IMPORT BACKGROUND MASK #######

    path = str(Path('preprocessing/background_mask.jpg').resolve())
    background_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    ###### CHAMFER MATCHING #######

    for img in cat_sal_images:
        roi, roi_top_left, roi_buttom_right = chamfer_matching(chamfer_templates, img, background_mask)

        top_left, buttom_right, img_class = template_matching(templates, roi, background_mask)

        img_rect = write_text(img, img_class)

        top_left = top_left + roi_top_left
        buttom_right = bottum_right + roi_buttom_right
        cv2.rectangle(img_rect, top_left, buttom_right, (0, 0, 255), 4)

        show_img(img_rect, 'Region of interest', wait_key=True)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
