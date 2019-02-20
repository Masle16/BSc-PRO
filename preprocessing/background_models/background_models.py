import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

filenames = glob.glob("/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/images_1280x720/baggrund/bevægelse/*.jpg")
images = [cv2.imread(img) for img in filenames]

avg = np.float32(images[0]) 

for img in images:
    cv2.accumulateWeighted(img, avg, 0.1)
    result = cv2.convertScaleAbs(avg)

#cv2.imshow("Average", result)

img = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/images_1280x720/kartofler/bevægelse/WIN_20190131_09_57_44_Pro.jpg')
#cv2.imshow('img', img)

diff = cv2.absdiff(result, img)

diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(diff_gray, 50, 255, 0)
diff_cnts, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(cnt) for cnt in cnts]
max_idx = np.argmax(areas)
cnt = cnts[max_idx]

x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(img, (x,y), (x+w, y+h), 255)
cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()