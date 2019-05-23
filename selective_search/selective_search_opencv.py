#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

""" Selective search opencv implementation """

from __future__ import division
import glob
# import sys
from pathlib import Path
import time
import numpy as np
import cv2

def main():
    """ Main function """

    # Load background mask
    path = str(Path('preprocessing/bgd_mask.jpg').resolve())
    bgd_mask = cv2.imread(path, cv2.IMREAD_COLOR)

    # # Load average background
    # path = str(Path('preprocessing/avg_background.jpg').resolve())
    # avg_bgd = cv2.imread(path, cv2.IMREAD_COLOR)

    # Image paths
    path_images = [
        str(Path('dataset3/res_still/test/background/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/potato/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/carrots/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/catfood_salmon/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/catfood_beef/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/bun/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/arm/*.jpg').resolve()),
        str(Path('dataset3/res_still/test/ketchup/*.jpg').resolve())
    ]

    # Speed-up using multithreads
    cv2.setUseOptimized(True)

    # Create selective search segmentation object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # Load image
    # img_fil = glob.glob(path_images[2])
    # img = cv2.imread(img_fil[3], cv2.IMREAD_ANYCOLOR)
    path = str(Path('/home/mathi/Downloads/potato.jpg').resolve())
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    dst = img.copy()

    # Remove unnessary background
    img = cv2.bitwise_and(img, bgd_mask)

    # Start counting time
    tic = time.time()

    # Set input image on which we will run segmentation
    ss.setBaseImage(img)

    # # Switch to fast but low recall selective search method
    # ss.switchToSelectiveSearchFast()

    # Switch to high recall but slow selective search method
    ss.switchToSelectiveSearchQuality()

    # Run selective search segmentation on input image
    results = ss.process()

    # Remove rect with certain area
    rects = []
    for x, y, w, h in results:
        area = (w * h)

        if area < 2500:
            continue

        if area > 160000:
            continue

        rects.append((x, y, w, h))

    # Print time spent on selective search
    toc = time.time()
    print('Time:', (toc - tic))

    # Print number of rects found
    print('Total number of region proposals:', len(rects))

    # Create a copy of original image
    overlay = dst.copy()
    output = dst.copy()

    # Iterate over all regions proposals
    for i, rect in enumerate(rects):
        if i < 3:
            # draw rectangle for region proposal
            x, y, w, h, = rect
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), -1)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 125, 0), 2)
        else:
            break

    cv2.addWeighted(overlay, 0.5, output, 1 - 0.5, 0, output)

    # Show output
    cv2.imshow('Output', output)
    cv2.waitKey(0)

    # Close image show windown
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
