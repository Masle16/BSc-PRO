import cv2
import numpy as np
from matplotlib import pyplot as plt

def temp_match_meth(template, src):
    """
    displays six methods for template matching and shows how the perform
    @template, the object you are searching for
    @src, the source image you are searching in
    """

    # convert to gray
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = template.shape[: : -1]

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 
                'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 
                'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = img_gray
        method = eval(meth)

        # Apply template matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        
        bottom_right = (top_left[0] + w, top_left[1] + h)

        print(method)

        cv2.rectangle(img, top_left, bottom_right, 255, 2)

        cv2.imshow('Matching Result', res)

        cv2.imshow('Detected Point', img)

        cv2.waitKey(0)

def template_matching(template, src, method=cv2.TM_SQDIFF):
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
    """

    # convert to gray
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[: : -1]

    # Display template
    #cv2.imshow("template", template)
    #cv2.waitKey(0)
    #cv2.destroyWindow("template")

    # Apply template matching
    result = cv2.matchTemplate(img_gray, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if (method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED):
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Display detected area
    #img = src.copy()
    #cv2.rectangle(img, top_left, bottom_right, 255, 2)
    #cv2.imshow("Detected point", img)
    #cv2.waitKey(0)
    #cv2.destroyWindow("Detected point")

    # Crop region of interest
    radius = 224
    x_ctr, y_ctr = int((top_left[0] + bottom_right[0]) / 2), int((top_left[1] + bottom_right[1]) / 2)
    x_left, x_right, y_up, y_down = x_ctr - radius, x_ctr + radius, y_ctr - radius, y_ctr + radius 

    if (x_right > src.shape[1]):
        margin = -1 * (src.shape[1] - x_right)
        x_right -= margin; x_left -= margin
    elif (x_left < 0):
        margin = -1 * x_left
        x_right += margin; x_left += margin

    if (y_up < 0):
        margin = -1 * y_up
        y_down += margin; y_up += margin
    elif (y_down > src.shape[0]):
        margin = -1 * (src.shape[0] - y_down)
        y_down -= margin; y_up -= margin

    img_crop = src[y_up : y_down, x_left : x_right]

    return img_crop

src = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/potato_and_catfood/train/potato/WIN_20190131_10_04_48_Pro (2).jpg')
template = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/template_matching/template_tm.jpg')

roi = template_matching(template, src)

print(roi.shape)

cv2.imshow("roi", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()