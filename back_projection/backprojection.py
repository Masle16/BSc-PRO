import cv2
import numpy as np
from matplotlib import pyplot as plt

origi_img = cv2.imread('potato_and_catfood/train/potato/WIN_20190131_09_59_39_Pro.jpg')
roi_img = cv2.imread('back_projection/template_bp.jpg')

hsv_original = cv2.cvtColor(origi_img, cv2.COLOR_RGB2HSV)
hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_RGB2HSV)
 
hue, saturation, value = cv2.split(hsv_roi)

# Histogram ROI
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)
 
# Filtering remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.filter2D(mask, -1, kernel)
_, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
 
mask = cv2.merge((mask, mask, mask))
result = cv2.bitwise_and(origi_img, mask)

# Morphological transformations
kernel = np.ones((12,12), np.uint8)
opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

cv2.imshow("Original", origi_img)
cv2.imshow("Roi", roi_img)
cv2.imshow("Opening", opening)

cv2.waitKey(0)
cv2.destroyAllWindows()