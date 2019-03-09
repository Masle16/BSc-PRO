""" Module for template matching and chamfer matching """

import glob
import cv2
import imutils
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def templateMatchMeth(template, src):
    """
    displays six methods for template matching and shows how the perform
    @template, the object you are searching for
    @src, the source image you are searching in
    """

    # convert to gray
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = template.shape[: : -1]

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
               'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
               'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = img_gray
        method = eval(meth)

        # Apply template matching
        res = cv2.matchTemplate(img, template, method)
        _, _, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        print(method)

        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        cv2.imshow('Matching Result', res)
        cv2.imshow('Detected Point', img)
        cv2.waitKey(0)

    return img

def multiscaleTemplateMatch(template, src, method=cv2.TM_SQDIFF):
    """
    returns crop image of interest
    @template, the template you are searching for
    @src, the source image you are searching in
    @method, the method you are using (preset to cv2.TM_SQDIFF)
        - cv2.TM_CCOEFF
        - cv2.TM_CCOEFF_NORMED
        - cv2.TM_CCORR
        - cv2.TM_CCORR_NORMED
        - cv2.TM_SQDIFF
        - cv2.TM_SQDIFF_NORMED)

    Uses multiscaling to scale image
    """

    # Make private copies of src and template
    _template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Store height and width
    height = template.shape[0]
    width = template.shape[1]

    # Resize template
    ratio = 224.0 / _template.shape[1]
    dim = (224, int(_template.shape[0] * ratio))
    _template = cv2.resize(_template, dim, interpolation=cv2.INTER_AREA)

    found = (None, None, None, None, None, None)
    for scale in np.linspace(1.0, 2.0, 5)[::-1]:

        # Resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = cv2.resize(_template,
                             (int(dim[0] * scale), int(dim[1] * scale)),
                             interpolation=cv2.INTER_AREA)

        # If the resized image is smaller than the template, then break from the loop
        if resized.shape[0] > _src.shape[0] or resized.shape[1] > _src.shape[1]:
            continue

        # Rotate image to different angles
        for angle in np.arange(0, 360, 45):
            rotated = imutils.rotate_bound(resized, angle)

            # Detect edges in the resized grayscale image and apply template
            # matching to find the template in the image
            result = cv2.matchTemplate(_src, rotated, method)
            (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(result)

            # #############
            # # Visualize #
            # #############
            # # Draw a bounding box around the detected region
            # clone = src.copy()
            # if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
            #     top_left = min_loc
            # else:
            #     top_left = max_loc
            # cv2.rectangle(clone,
            #               (top_left[0], top_left[1]),
            #               (top_left[0] + resized.shape[0], top_left[1] + resized.shape[1]),
            #               (0, 0, 255),
            #               2)
            # cv2.imshow('Visualize', clone)
            # cv2.imshow('Template', rotated)
            # cv2.waitKey(0)

            # If we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found[0] is None:
                found = (min_val, max_val, min_loc, max_loc, resized.shape[0], resized.shape[1])

            if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
                if min_val < found[0]:
                    found = (min_val, max_val, min_loc, max_loc, resized.shape[0], resized.shape[1])
            else:
                if max_val > found[1]:
                    found = (min_val, max_val, min_loc, max_loc, resized.shape[0], resized.shape[1])

    # Unpack the found variable and compute the (x,y) coordinates of the bounding
    # rect based of the resized ratio
    (min_val, max_val, min_loc, max_loc, width, height) = found
    if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + width, top_left[1] + height)

    # ################################
    # # Display bounding rect chosen #
    # ################################
    # clone = src.copy()
    # cv2.rectangle(clone,
    #               (top_left[0], top_left[1]),
    #               (top_left[0] + width, top_left[1] + height),
    #               (0, 0, 255),
    #               2)
    # cv2.imshow('Visualize', clone)

    # Crop region of interest
    radius = 224
    x_ctr = int((top_left[0] + bottom_right[0]) / 2)
    y_ctr = int((top_left[1] + bottom_right[1]) / 2)
    x_left = x_ctr - radius
    x_right = x_ctr + radius
    y_up = y_ctr - radius
    y_down = y_ctr + radius

    if x_right > src.shape[1]:
        margin = -1 * (src.shape[1] - x_right)
        x_right -= margin
        x_left -= margin
    elif x_left < 0:
        margin = -1 * x_left
        x_right += margin
        x_left += margin

    if y_up < 0:
        margin = -1 * y_up
        y_down += margin
        y_up += margin
    elif y_down > src.shape[0]:
        margin = -1 * (src.shape[0] - y_down)
        y_down -= margin
        y_up -= margin

    # Display detected area
    img_rect = src.copy()
    cv2.rectangle(img_rect, (x_left, y_up), (x_right, y_down), (0, 0, 255), 4)
    cv2.imshow('Bounding rect', img_rect)
    cv2.waitKey(0)
    #cv2.imwrite('/home/mathi/Desktop/img_rect.jpg', img_rect)

    img_crop = src[y_up : y_down, x_left : x_right]

    return img_crop

def chamferMatch(template, src, method=cv2.TM_SQDIFF):
    """
    returns crop image of interest
    @template, the distance map of the template you are searching for
    @src, the source image you are searching in
    @method, the method you are using (preset to cv2.TM_SQDIFF)
        - cv2.TM_CCOEFF
        - cv2.TM_CCOEFF_NORMED
        - cv2.TM_CCORR
        - cv2.TM_CCORR_NORMED
        - cv2.TM_SQDIFF
        - cv2.TM_SQDIFF_NORMED)
    """

    # store template width and height
    width, height = template.shape[::-1]

    # Convert to src to distance map
    _chamfer_src = cv2.Canny(src, 100, 200)
    _, _chamfer_src = cv2.threshold(_chamfer_src, 127, 255, cv2.THRESH_BINARY_INV)
    img_dist = cv2.distanceTransform(_chamfer_src, cv2.DIST_L2, 3)

    # img_dist_norm = cv2.normalize(img_dist,
    #                               None,
    #                               0,
    #                               200,
    #                               norm_type=cv2.NORM_MINMAX)

    # cv2.imwrite('/home/mathi/Desktop/chamfer_input_image.jpg', img_dist_norm)

    # Apply template matching
    result = cv2.matchTemplate(img_dist, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # match_space = cv2.normalize(result,
    #                             None,
    #                             0,
    #                             255,
    #                             norm_type=cv2.NORM_MINMAX)

    # # cv2.imwrite('/home/mathi/Desktop/match_space.jpg', match_space)

    # """ Surface plot """
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.arange(0, match_space.shape[0], 1)
    # y = np.arange(0, match_space.shape[1], 1)
    # X, Y = np.meshgrid(x, y)
    # zs = np.array([match_space[x, y] for x, y in zip(np.ravel(X), np.ravel(Y))])
    # Z = zs.reshape(X.shape)

    # ax.plot_surface(X, Y, Z, cmap=cm.jet)

    # ax.set_xlabel('Row')
    # ax.set_ylabel('Column')
    # ax.set_zlabel('Grayscale')

    # plt.show()

    if method is cv2.TM_SQDIFF or method is cv2.TM_SQDIFF_NORMED:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + width, top_left[1] + height)

    # Crop region of interest
    radius = 224
    x_ctr = int((top_left[0] + bottom_right[0]) / 2)
    y_ctr = int((top_left[1] + bottom_right[1]) / 2)
    x_left = x_ctr - radius
    x_right = x_ctr + radius
    y_up = y_ctr - radius
    y_down = y_ctr + radius

    if x_right > src.shape[1]:
        margin = -1 * (src.shape[1] - x_right)
        x_right -= margin
        x_left -= margin
    elif x_left < 0:
        margin = -1 * x_left
        x_right += margin
        x_left += margin

    if y_up < 0:
        margin = -1 * y_up
        y_down += margin
        y_up += margin
    elif y_down > src.shape[0]:
        margin = -1 * (src.shape[0] - y_down)
        y_down -= margin
        y_up -= margin

    # Display detected area
    img_rect = src.copy()
    cv2.rectangle(img_rect, (x_left, y_up), (x_right, y_down), (0, 0, 255), 4)
    cv2.imwrite('/home/mathi/Desktop/img_rect.jpg', img_rect)

    img_crop = src[y_up : y_down, x_left : x_right]

    return img_crop

def main():
    """ Main function """

    template = cv2.imread(
        '/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/preprocessing/template_matching/template_tm2.jpg'
    )

    # Check template
    if template is None:
        raise Exception('Template equals None')

    chamfer_template = cv2.Canny(template, 100, 200)
    _, chamfer_template = cv2.threshold(chamfer_template, 127, 255, cv2.THRESH_BINARY_INV)
    chamfer_template = cv2.distanceTransform(chamfer_template, cv2.DIST_L2, 3)

    tmp_dist_norm = cv2.normalize(chamfer_template,
                                  None,
                                  0,
                                  255,
                                  norm_type=cv2.NORM_MINMAX)

    cv2.imwrite('/home/mathi/Desktop/tmp_dist.jpg', tmp_dist_norm)

    potato_fil = glob.glob(
        '/mnt/sdb1/Robtek/6semester/Bachelorproject/BSc-PRO/potato_and_catfood/train/potato/*.jpg'
    )
    potato_images = [cv2.imread(img) for img in potato_fil]

    chamferMatch(chamfer_template, potato_images[1])

    # d = 0
    # for img in potato_images:

    #     multiscaleTemplateMatch(template, img)
    #     chamferMatch(chamfer_template, img)

    #     # path =   '/mnt/sdb/Robtek/6semester/Bachelorproject/BSc-PRO/\
    #     #           preprocessing/template_matching/cropped_potatoes_cm/potato_%d.jpg' %d
    #     # cv2.imwrite(path, roi)
    #     # d += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
