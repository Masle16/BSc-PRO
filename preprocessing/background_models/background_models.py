import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

def run_avg(path):
    """
    returns running average of all images in path folder
    @path, a string of the path to the folder with images (jpg)
    """

    path_img = path + '/*jpg'
    files = glob.glob(path_img)
    file_images = [cv2.imread(img) for img in files]
    
    avg = np.float32(file_images[0])

    for img in file_images:
        cv2.accumulateWeighted(img, avg, 0.1)
        
    result = cv2.convertScaleAbs(avg)
    #cv2.imshow('result', result)
    #cv2.waitKey(0)

    return result

def background_sub(img, background):
    """
    returns cropped image(448 x 448) of region of interest
    @img, the image of interest
    """

    # Calculate image difference and find largest contour
    diff = cv2.absdiff(background, img)

    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(diff_gray, 50, 255, 0)
    diff_cnts, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(cnt) for cnt in cnts]
    max_idx = np.argmax(areas)
    cnt = cnts[max_idx]

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

    # Get region of interest
    roi = img[y_up : y_down, x_left : x_right]

    return roi

"""
potato_fil = glob.glob('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/potato_and_catfood/train/potato/*.jpg')
potato_images = [cv2.imread(img) for img in potato_fil]

background_img = run_avg('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/images_1280x720/baggrund/bevægelse')

d = 0
for img in potato_images:
    roi = background_sub(img, background_img)
    path = '/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/background_models/cropped_potatoes/potato_%d.jpg' %d
    cv2.imwrite(path, roi)
    d += 1

cv2.destroyAllWindows()
"""