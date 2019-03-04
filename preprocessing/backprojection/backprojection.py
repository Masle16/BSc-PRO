""" Module for Back-projection """

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

def backproject(roi_hist, img):
    """
    returns backprojected image
    @roi_hist, histogram of region of interest to find
    @img, image to search in
    """

    # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create Histogram of roi and create mask from the histogram
    mask = cv2.calcBackProject([img_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    cv2.imwrite('/home/mathi/Desktop/bp_img.jpg', mask)

    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.filter2D(mask, -1, kernel)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    mask = cv2.merge((mask, mask, mask))
    result = cv2.bitwise_and(img, mask)

    kernel = np.ones((12, 12), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    # Find contours
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 0, 127, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    #cv2.imshow("Contours", img)
    #cv2.waitKey(0)

    # Find biggest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    # Crop contour form image
    x, y, w, h = cv2.boundingRect(cnt)
    x_ctr, y_ctr = int((x + (x + w)) / 2), int((y + (y + h)) / 2)
    radius = 224
    x_left, x_right, y_up, y_down = x_ctr - radius, x_ctr + radius, y_ctr - radius, y_ctr + radius 

    if (x_right > img.shape[1]):
        margin = -1 * (img.shape[1] - x_right)
        x_right -= margin; x_left -= margin
    elif (x_left < 0):
        margin = -1 * x_left
        x_right += margin; x_left += margin

    if (y_up < 0):
        margin = -1 * y_up
        y_down += margin; y_up += margin
    elif (y_down > img.shape[0]):
        margin = -1 * (img.shape[0] - y_down)
        y_down -= margin; y_up -= margin

    img_crop = img[y_up : y_down, x_left : x_right]

    cv2.imwrite('/home/mathi/Desktop/crop_img.jpg', img_crop)

    return img_crop

def main():
    """ Main function """

    roi_img = cv2.imread('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/backprojection/template_bp.jpg')
    roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    cv2.imwrite('/home/mathi/Desktop/template.jpg', roi_img)
    cv2.imwrite('/home/mathi/Desktop/hsv_template.jpg', roi_hsv)
    cv2.imwrite('/home/mathi/Desktop/hist_template.jpg', roi_hist)

    potato_fil = glob.glob('/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/potato_and_catfood/train/potato/*.jpg')
    potato_images = [cv2.imread(img) for img in potato_fil]

    cv2.imwrite('/home/mathi/Desktop/input_img.jpg', potato_images[0])
    roi = backproject(roi_hist, potato_images[0])
    cv2.imshow('Original image', potato_images[0])
    cv2.imshow('Region of interest', roi)
    cv2.waitKey(0)

    # d = 0
    # for img in potato_images:
    #     roi = backproject(roi_hist, img)
    #     #roi = get_item(roi, img)

    #     cv2.imshow('Original image', img)
    #     cv2.imshow('Region of interest', roi)
    #     cv2.waitKey(0)
        
    #     #path = '/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/back-projection/potatoes/potato_%d.jpg' %d
    #     #cv2.imwrite(path, roi)

    #     d += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
