""" Module for template matching and chamfer matching """

import glob
import cv2
import imutils
import numpy as np

def show_img(img, window_name, width=640, height=400, wait_key=False):
    """ Show image in certain size """

    resized = cv2.resize(img,
                         (width, height),
                         interpolation=cv2.INTER_NEAREST)

    cv2.imshow(window_name, resized)

    if wait_key is True:
        cv2.waitKey(0)

    return 0

def remove_background():
    """ returns image with no background, only table """

    # Get background
    path = '/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/images_1280x720/baggrund/bevægelse/WIN_20190131_10_31_36_Pro.jpg'
    background = cv2.imread(path, cv2.IMREAD_COLOR)

    # Find background pixels coordinates
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 64), (179, 51, 255))
    result = cv2.bitwise_and(background, background, mask=mask)

    return mask, result

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

def template_matching(template, src, method=cv2.TM_CCORR_NORMED):
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

    # Make private copies of src and template
    _template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Store height and width
    height = template.shape[0]
    width = template.shape[1]

    # Resize template
    ratio = 224.0 / _template.shape[1]
    dim = (224, int(_template.shape[0] * ratio))
    _template = cv2.resize(_template, dim, interpolation=cv2.INTER_AREA)

    found = None
    for scale in np.linspace(1.0, 2.0, 5)[::-1]:

        # Resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = cv2.resize(_template,
                             (int(dim[0] * scale), int(dim[1] * scale)),
                             interpolation=cv2.INTER_AREA)

        # If the resized image is smaller than the template, then break from the loop
        if resized.shape[0] > _src.shape[0] or resized.shape[1] > _src.shape[1]:
            continue

        # Rotate image to different angles
        for angle in np.arange(0, 360, 45):
            rotated = imutils.rotate_bound(resized, angle)

            # Detect edges in the resized grayscale image and apply template
            # matching to find the template in the image
            result = cv2.matchTemplate(_src, rotated, method)
            (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(result)

            # #############
            # # Visualize #
            # #############
            # # Draw a bounding box around the detected region
            # clone = src.copy()
            # if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
            #     top_left = min_loc
            # else:
            #     top_left = max_loc
            # cv2.rectangle(clone,
            #               (top_left[0], top_left[1]),
            #               (top_left[0] + resized.shape[0], top_left[1] + resized.shape[1]),
            #               (0, 0, 255),
            #               2)
            # cv2.imshow('Visualize', clone)
            # cv2.imshow('Template', rotated)
            # cv2.waitKey(0)

            # If we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None:
                found = (min_val, max_val, min_loc, max_loc, resized.shape[0], resized.shape[1])

            if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
                if min_val < found[0]:
                    found = (min_val, max_val, min_loc, max_loc, resized.shape[0], resized.shape[1])
            else:
                if max_val > found[1]:
                    found = (min_val, max_val, min_loc, max_loc, resized.shape[0], resized.shape[1])

    # Unpack the found variable and compute the (x,y) coordinates of the bounding
    # rect based of the resized ratio
    (min_val, max_val, min_loc, max_loc, height, width) = found
    if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + width, top_left[1] + height)

    # ################################
    # # Display bounding rect chosen #
    # ################################
    # clone = src.copy()
    # cv2.rectangle(clone,
    #               (top_left[0], top_left[1]),
    #               (top_left[0] + width, top_left[1] + height),
    #               (0, 0, 255),
    #               2)
    # cv2.imshow('Visualize', clone)

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

    img_crop = src[y_up : y_down, x_left : x_right]

    return img_crop

def chamfer_matching(templates, src, method=cv2.TM_SQDIFF):
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
    _mask, _ = remove_background()
    img = cv2.bitwise_and(src, src, mask=_mask)

    # Convert to src to distance map
    _chamfer_src = cv2.Canny(img, 100, 200)
    _, _chamfer_src = cv2.threshold(_chamfer_src, 127, 255, cv2.THRESH_BINARY_INV)
    img_dist = cv2.distanceTransform(_chamfer_src, cv2.DIST_L2, 3)

    img_rect = img.copy()
    found = None
    for template in templates:

        # Apply template matching
        matching_space = cv2.matchTemplate(img_dist, template, method)

        # Find local maximum
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching_space)

        # If we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None:
            found = (min_val, max_val, min_loc, max_loc, template.shape[0], template[1])

        if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
            if min_val < found[0]:
                found = (min_val, max_val, min_loc, max_loc, template.shape[0], template.shape[1])
        else:
            if max_val > found[1]:
                found = (min_val, max_val, min_loc, max_loc, template.shape[0], template.shape[1])

        cv2.rectangle(img_rect, min_loc,
                      (min_loc[0] + template.shape[0], min_loc[1] + template.shape[1]),
                      (0, 0, 255),
                      4)

        show_img(img_rect, 'Detected area', wait_key=True)

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

    img_crop = src[y_up : y_down, x_left : x_right]

    return img_crop

def main():
    """ Main function """

    template_potato = cv2.imread('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/template_matching/template_potato.jpg', cv2.IMREAD_COLOR)
    template_carrot = cv2.imread('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/template_matching/template_carrot.jpg', cv2.IMREAD_COLOR)
    template_cat_beef = cv2.imread('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/template_matching/template_cat_beef.jpg', cv2.IMREAD_COLOR)
    template_cat_sal = cv2.imread('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/template_matching/template_cat_sal.jpg', cv2.IMREAD_COLOR)

    templates = [template_potato, template_carrot, template_cat_beef, template_cat_sal]

    chamfer_templates = []
    for temp in templates:

        # Check template
        if temp is None:
            raise Exception('Template equals None')

        # Create chamfer template
        chamfer = cv2.Canny(temp, 100, 200)
        _, chamfer = cv2.threshold(chamfer, 125, 255, cv2.THRESH_OTSU)
        chamfer = cv2.distanceTransform(chamfer, cv2.DIST_L2, 3)
        chamfer_templates.append(chamfer)

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

    for img in potato_images:
        chamfer_matching(chamfer_templates, img)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
