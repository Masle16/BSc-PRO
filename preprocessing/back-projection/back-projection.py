import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

def backproject(roi, img):
    """
    returns backprojected image
    @roi, region of interest to find
    @img, image to search in
    """

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

def get_item(img, origi_img):
    """
    returns 224 x 224 image of interesting contour
    @img: backprojected image
    @origi_img: original image
    """

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
    x_ctr, y_ctr = int((x + (x + w)) / 2), int((y + (y + h)) / 2)
    radius = 224
    x_left, x_right, y_up, y_down = x_ctr - radius, x_ctr + radius, y_ctr - radius, y_ctr + radius 

    if (x_right > origi_img.shape[1]):
        margin = -1 * (origi_img.shape[1] - x_right)
        x_right -= margin; x_left -= margin
    elif (x_left < 0):
        margin = -1 * x_left
        x_right += margin; x_left += margin

    if (y_up < 0):
        margin = -1 * y_up
        y_down += margin; y_up += margin
    elif (y_down > origi_img.shape[0]):
        margin = -1 * (origi_img.shape[0] - y_down)
        y_down -= margin; y_up -= margin

    img_crop = origi_img[y_up : y_down, x_left : x_right]

    return img_crop

roi_img = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/back-projection/template_bp.jpg')

potato_fil = glob.glob('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/potato_and_catfood/train/potato/*.jpg')
potato_images = [cv2.imread(img) for img in potato_fil]

d = 0
for img in potato_images:
    roi = backproject(roi_img, img)
    roi = get_item(roi, img)
    cv2.imshow("Roi", roi)
    cv2.waitKey(0)

cv2.destroyAllWindows()