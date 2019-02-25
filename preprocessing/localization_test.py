import cv2
import numpy as np
import glob

# Import data 
potato_fil = glob.glob('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/potato_and_catfood/train/potato/*.jpg')
potato_images = [cv2.imread(img) for img in potato_fil]

from background_models import background_models as bm
from template_matching import template_matching as tm
from backprojection import backprojection as bp

# Running average of background image
background_img = bm.run_avg('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/images_1280x720/baggrund/bevægelse')

# Create histogram of template
roi_img = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/backprojection/template_bp.jpg')
roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# Create distance map of template
template = cv2.imread('/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/template_matching/template_tm2.jpg')
tmp_bw = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
_, tmp_bw = cv2.threshold(tmp_bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
tmp_dist = cv2.distanceTransform(tmp_bw, cv2.DIST_L2, 3)

for img in potato_images:

    # Create region of interest for each method
    roi_bm = bm.background_sub(img, background_img) # Background models
    roi_bp = bp.backproject(roi_hist, img)          # Back-projection
    roi_tm = tm.templateMatch(template, img)        # Template matching
    roi_cm = tm.chamferMatch(tmp_dist, img)

    # Display images
    cv2.imshow('Background models', roi_bm)
    cv2.imshow('Back-projection', roi_bp)
    cv2.imshow('Template matching', roi_tm)
    cv2.imshow('Chamfer matchibg', roi_cm)
    cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()