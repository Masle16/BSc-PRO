import cv2
import numpy as np
from matplotlib import pyplot as plt

bgr_origi_img = cv2.imread('back_projection/potato.jpg')
bgr_roi_img = cv2.imread('back_projection/template_bp.jpg')

b, g, r = cv2.split(bgr_origi_img)
original_image = cv2.merge([r, g, b])

b, g, r = cv2.split(bgr_roi_img)
roi = cv2.merge([r, g, b])

hsv_original = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
 
hue, saturation, value = cv2.split(hsv_roi)

# Histogram ROI
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)
 
# Filtering remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.filter2D(mask, -1, kernel)
_, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
 
mask = cv2.merge((mask, mask, mask))
result = cv2.bitwise_and(original_image, mask)

# Morphological transformations
kernel = np.ones((12,12), np.uint8)
opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

cv2.imshow("test", opening)

# make figure
fig = plt.figure(figsize=(9,8))

fig.add_subplot(2, 2, 1)
plt.title("original image")
plt.imshow(original_image)

fig.add_subplot(2, 2, 2)
plt.title("roi")
plt.imshow(roi)

fig.add_subplot(2, 2, 3)
plt.title("mask")
plt.imshow(mask)

fig.add_subplot(2, 2, 4)
plt.title("result")
plt.imshow(opening)

# show
plt.show()