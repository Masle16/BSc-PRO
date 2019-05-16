""" Selective search opencv implementation """

from __future__ import division
import glob
# import sys
from pathlib import Path
import cv2

def main():
    """ Main function """

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
    cv2.setNumThreads(4)

    # read image
    img_fil = glob.glob(path_images[7])
    img = cv2.imread(img_fil[0], cv2.IMREAD_ANYCOLOR)

    # # Resize image
    # new_h = 200
    # new_w = int((img.shape[1] * 200) / img.shape[0])
    # img = cv2.resize(img, (new_w / new_h))

    # Create selective search segmentation object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSerachSegmentation()

    # Set input image on which we will run segmentation
    ss.setBaseImage(img)

    # Run selective search segmentation on input image
    rects = ss.process()
    print('Total number of region proposals:', len(rects))

    # Number of region proposals to show
    num_show_rects = 100

    # Increment to increase/descrease total number of reason proposals to be shown
    # increment = 50

    while True:
        # Create a copy of original image
        img_out = img.copy()

        # Iterate over all regions proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till num_show_rects
            if i < num_show_rects:
                x, y, w, h, = rect
                cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        # Show output
        cv2.imshow('Output', img_out)
        cv2.waitKey(0)

    # Close image show windown
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
