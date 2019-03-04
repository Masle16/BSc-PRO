""" Module for template matching and chamfer matching """

import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

def templateMatchMeth(template, src):
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

def templateMatch(template, src, method=cv2.TM_SQDIFF):
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

def multiscaleTemplateMatch(template, src, method=cv2.TM_CCOEFF):
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

    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    
    found = None
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # Resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = cv2.resize(img_gray, (int(img_gray.shape[1] * scale), img_gray.shape[0]))
        r = img_gray.shape[1] / float(resized.shape[1])

        # If the resized image is smaller than the template, then break from the loop
        if resized.shape[0] < h or resized.shape[1] < w:
            break

        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template_gray, method)
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(result)
        
        if found == None or max_val > found[0]:
            found = (max_val, max_loc, r)

    # Unpack the found variable and compute the (x,y) coordinates of the bounding
    # rect based of the resized ratio
    (max_val, max_loc, r) = found
    if (method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED):
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)

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

def chamferMatch(template, src, method=cv2.TM_SQDIFF):
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

    # store template width and height
    w, h = template.shape[::-1]

    # Convert to src to distance map 
    img_bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, img_bw = cv2.threshold(img_bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_dist = cv2.distanceTransform(img_bw, cv2.DIST_L2, 3)

    # Apply template matching
    result = cv2.matchTemplate(img_dist, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if (method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED):
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)

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

def main():
    template = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/template_matching/template_tm2.jpg')
    tmp_bw = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, tmp_bw = cv2.threshold(tmp_bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    tmp_dist = cv2.distanceTransform(tmp_bw, cv2.DIST_L2, 3)

    potato_fil = glob.glob('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/potato_and_catfood/train/potato/*.jpg')
    potato_images = [cv2.imread(img) for img in potato_fil]

    d = 0
    for img in potato_images:
        roi_cm = chamferMatch(tmp_dist, img)
        roi_tm = templateMatch(template, img)

        cv2.imshow('Original image', img)
        cv2.imshow('Chamfer matching', roi_cm)
        cv2.imshow('Template matching', roi_tm)
        cv2.waitKey(0)

        #path = '/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/template_matching/cropped_potatoes_cm/potato_%d.jpg' %d
        #cv2.imwrite(path, roi)
        
        d += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()