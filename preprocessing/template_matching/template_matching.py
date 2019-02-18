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
    cv2.rectangle(src, top_left, bottom_right, 255, 2)
    cv2.imshow("Detected point", src)
    cv2.waitKey(0)
    cv2.destroyWindow("Detected point")

    img_crop = src[top_left[1] : top_left[1] + h, top_left[0] : top_left[0] + w]
    return img_crop

src = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/potato_and_catfood/train/potato/WIN_20190131_10_04_48_Pro (2).jpg')
template = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/template_matching/template_tm.jpg')

roi = template_matching(template, src)

print(roi.shape)

cv2.imshow("roi", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()