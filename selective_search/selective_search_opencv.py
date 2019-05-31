#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

""" Selective search opencv implementation """

from __future__ import division
import glob
# import sys
from pathlib import Path
import time
import numpy as np
import cv2

#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlap_thresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

		# loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

			# if there is sufficient overlap, suppress the
			# current bounding box
            if overlap > overlap_thresh:
                suppress.append(pos)

		# delete all indexes from the index list that are in the
		# suppression list
        idxs = np.delete(idxs, suppress)

	# return only the bounding boxes that were picked
    return boxes[pick]

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
    path = str(Path('selective_search/input.png').resolve())
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    output = img.copy()

    # Remove unnessary background
    img = cv2.bitwise_and(img, bgd_mask)

    # Start counting time
    tic = time.time()

    # Set input image on which we will run segmentation
    ss.setBaseImage(img)

    # # Switch to fast but low recall selective search method
    ss.switchToSelectiveSearchFast()

    # Switch to high recall but slow selective search method
    # ss.switchToSelectiveSearchQuality()

    # Run selective search segmentation on input image
    results = ss.process()

    # # Remove rect with certain area
    # rects = []
    # for x, y, w, h in results:
    #     area = (w * h)

    #     if area < 2500:
    #         continue

    #     if area > 160000:
    #         continue

    #     bbox = np.array([x, y, x+w, y+h])
    #     rects.append(bbox)

    # Non maxima suppression
    pick = non_max_suppression_slow(results, 0.3)
    for (start_x, start_y, end_x, end_y) in pick:
        cv2.rectangle(output,
                      (start_x, start_y),
                      (end_x, end_y),
                      (0, 125, 0),
                      2)  

    # Print time spent on selective search
    toc = time.time()
    print('Time:', (toc - tic))

    # Print number of rects found
    print('Total number of region proposals:', len(pick))

    # # Iterate over all regions proposals
    # for i, rect in enumerate(rects):
    #     if i < 100:
    #         # draw rectangle for region proposal
    #         x, y, w, h, = rect
    #         cv2.rectangle(output, (x, y), (x+w, y+h), (0, 125, 0), 2)
    #     else:
    #         break

    # Display rects in output image
    cv2.imshow('Rects', output)
    cv2.waitKey()

    # Close image show windown
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
