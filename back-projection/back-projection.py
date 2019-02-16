import cv2
import numpy as np
from matplotlib import pyplot as plt

#############################################################
#               Back-projection function                    #
#                                                           #
#   - roi:      template image                              #
#   - img:      image to search in                          #
#   - return:   Backprojected image                         #
#                                                           #
#############################################################
def backproject(roi, img):
    # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Extract hue, saturation and value
    hue, saturation, value = cv2.split(roi_hsv)

    # Create Histogram of roi and create mask from the histogram
    roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    mask = cv2.calcBackProject([img_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.filter2D(mask, -1, kernel)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    mask = cv2.merge((mask, mask, mask))
    result = cv2.bitwise_and(img, mask)

    kernel = np.ones((12,12), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    # Display result
    #cv2.imshow("Result", result)
    #cv2.waitKey(0)

    return result

#############################################################
#       Get bounding rect of largest contour                #
#                                                           #
#   - img: backporjected image                              #
#   - origi_img: original image                             #
#   - return: cropped image of biggest contour              #
#                                                           #
#############################################################
def get_item(img, origi_img):
    # Find contours
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 0, 127, 0)
    img_contours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
    img_crop = origi_img[y:y+h, x:x+w]
    img_crop = cv2.resize(img_crop, (224, 224))

    # Store image
    cv2.imwrite('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/back-projection/potatoes/potato.jpg', img_crop, [int(cv2.IMWRITE_JPEG_OPTIMIZE), 120])

    return img_crop

origi_img = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/potato_and_catfood/train/potato/WIN_20190131_10_00_14_Pro.jpg')
roi_img = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/back-projection/template_bp.jpg')

img = backproject(roi_img, origi_img)

img = get_item(img, origi_img)