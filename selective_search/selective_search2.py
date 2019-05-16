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

    # # Load image
    # img_fil = glob.glob(path_images[7])
    # img = cv2.imread(img_fil[0], cv2.IMREAD_ANYCOLOR)

    times = []
    rectangles = []

    for path in path_images:
        # Load images
        img_fil = glob.glob(path)
        images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in img_fil]

        for img in images:
            # Remove unnessary background
            img = cv2.bitwise_and(img, bgd_mask)

            # Set input image on which we will run segmentation
            ss.setBaseImage(img)

            # Start counting time
            tic = time.time()

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

                if area < 1550:
                    continue

                if area > 160000:
                    continue

                rects.append((x, y, w, h))

            # Print time spent on selective search
            toc = time.time()
            times.append((toc - tic))
            print('Time:', (toc - tic))

            # Print number of rects found
            rectangles.append(len(rects))
            print('Total number of region proposals:', len(rects))

    print('\nAverage time:', np.mean(times))
    print('\nAverage number of region proposals:', np.mean(rectangles))

    # # Create a copy of original image
    # img_out = img.copy()

    # # Iterate over all regions proposals
    # for i, rect in enumerate(rects):
    #     if i < 100:
    #         # draw rectangle for region proposal
    #         x, y, w, h, = rect
    #         cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 255, 0), 2, cv2.LINE_AA)
    #     else:
    #         break

    # # Show output
    # cv2.imshow('Output', img_out)
    # cv2.waitKey(0)

    # Close image show windown
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
