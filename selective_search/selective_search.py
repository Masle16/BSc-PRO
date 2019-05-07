#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

""" Module for Selective Search """

###### IMPORTS ######
from __future__ import division
import glob
from pathlib import Path
import time
import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

###### GLOBAL VARIABLES ######

###### FUNCTIONS ######
def generate_segments(src, scale, sigma, min_size):
    """ Segment smallest regions """

    # Open the image
    img_mask = skimage.segmentation.felzenszwalb(
        image=skimage.util.img_as_float(src),
        scale=scale,
        sigma=sigma,
        min_size=min_size
    )

    # Merge mask channel to the image as a 4th channel
    src = np.append(
        src, np.zeros(src.shape[:2])[:, :, np.newaxis], axis=2
    )
    src[:, :, 3] = img_mask

    return src

def sim_color(r1, r2):
    """ Calculate the sum of histogram intersection of color """

    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])

def sim_texture(r1, r2):
    """ Calculate the sum of histogram intersection of color """

    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])

def sim_size(r1, r2, img_size):
    """ Calculate the size similarity over the image """

    return 1.0 - (r1["size"] + r2["size"]) / img_size

def sim_fill(r1, r2, img_size):
    """ Calculate the fill similarity over the image """

    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"])) *
        (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )

    return 1.0 - (bbsize - r1["size"] - r2["size"]) / img_size

def calc_sim(r1, r2, img_size):
    """ Calculate the similarity over the image """

    return (
        sim_color(r1, r2) +
        sim_texture(r1, r2) +
        sim_size(r1, r2, img_size) +
        sim_fill(r1, r2, img_size)
    )

def calc_color_hist(img):
    """
    Calculate the color histogram fro each region

    The size of the output histogram will be BINS * COLOR_CHANNELS(3)

    Number of bins is 25

    Extract HSV
    """

    BINS = 25
    hist = np.array([])

    for color_channel in (0, 1, 2):
        # Extracting one color channel
        c = img[:, color_channel]

        # Calculate histogram for each color and join result
        hist = np.concatenate(
            [hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]]
        )

    # L1 normalize
    hist = hist / len(img)

    return hist

def calc_texture_gradient(img):
    """
    Calculate texture gradient for entire image

    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we use LBP instead

    Output will be [height(*)][width(*)]
    """

    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for color_channel in (0, 1, 2):
        ret[:, :, color_channel] = skimage.feature.local_binary_pattern(
            image=img[:, :, color_channel],
            P=8,
            R=1.0
        )

    return ret

def calc_texture_hist(img):
    """
    Calculate texture histogram for each region

    Calculate the histogram of gradient for each colors
    the size of output histogram will be:
        BINS * ORIENTATIONS * COLOR_CHANNELS(3)
    """

    BINS = 10
    hist = np.array([])

    for color_channel in (0, 1, 2):
        # Mask by the color channel
        fd = img[:, color_channel]

        # Calculate histogram for each orientation and
        # concatenate them all and join to the result
        hist = np.concatenate(
            [hist] + [np.histogram(a=fd, bins=BINS, range=(0.0, 1.0))[0]]
        )

    # L1 normalize
    hist = hist / len(img)

    return hist

def extract_regions(img):
    """ Extract regions """

    R = {}

    # Get hsv image
    hsv = skimage.color.rgb2hsv(img[:, :, :3])

    # Pass 1: count pixel positions
    for y, i in enumerate(img):
        for x, (r, g, b, l) in enumerate(i):
            # Initialize a new region
            if l not in R:
                R[l] = {
                    "min_x": 0xffff,
                    "min_y": 0xffff,
                    "max_x": 0,
                    "max_y": 0,
                    "labels": [l]
                }

            # Bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x

            if R[l]["min_y"] > y:
                R[l]["min_y"] = y

            if R[l]["max_x"] < x:
                R[l]["max_x"] = x

            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

    # Pass 2: calculate texture gradient
    tex_grad = calc_texture_gradient(img)

    # Pass 3: calculate color histogram of each region
    for k, v in list(R.items()):
        # Color histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = calc_color_hist(masked_pixels)

        # Texture histogram
        R[k]["hist_t"] = calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])

    return R

def extract_neighbours(region):
    """ Extracts neighbours """

    def intersect(a, b):
        if a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"]:
            return True

        if a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]:
            return True

        if a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]:
            return True

        if a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"]:
            return True

        # if (a["min_x"] < b["min_x"] < a["max_x"]
        #         and a["min_y"] < b["min_y"] < a["max_y"]) or (
        #             a["min_x"] < b["max_x"] < a["max_x"]
        #             and a["min_y"] < b["max_y"] < a["max_y"]) or (
        #                 a["min_x"] < b["min_x"] < a["max_x"]
        #                 and a["min_y"] < b["max_y"] < a["max_y"]) or (
        #                     a["min_x"] < b["max_x"] < a["max_x"]
        #                     and a["min_y"] < b["min_y"] < a["max_y"]):

        #     return True

        return False

    R = list(region.items())
    neighbors = []

    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbors.append((a, b))

    return neighbors

def merge_regions(r1, r2):
    """ Merges regions """

    new_size = r1["size"] + r2["size"]

    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }

    return rt

def selective_search(src, scale=1.0, sigma=0.8, min_size=50):
    """
    Selective Search

    Parameters
    ----------------
        src : ndarray
            Input image
        scale : int
            Free parameter.
            Higher means larges clusters in felzenzwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenswalb segmentation
        min_size : int
            Minimum component size for felzenswalb segmentation
    Returns
    ----------------
        img : ndarray
            Image with region label
            Region label is stored in the 4th value of each pixel [r, g, b, (region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                }
            ],
            ...
    """

    assert src.shape[2] == 3, "3ch image is expected"

    # Load image and get smallet regions
    # Region label is sored in the 4th value of each pixel [r, g, b, (region)]
    img = generate_segments(src, scale, sigma, min_size)

    if img is None:
        return None, {}

    img_size = img.shape[0] * img.shape[1]
    R = extract_regions(img)

    # Extract neighbouring information
    neighbors = extract_neighbours(R)

    # Calculate inital similarities
    S = {}
    for (ai, ar), (bi, br) in neighbors:
        S[(ai, bi)] = calc_sim(ar, br, img_size)

    # Hierachel search
    while S != {}:
        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Merge corresponding regions
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Mark similarites for regions to be removed
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)

        # Remove old similarities of related regions
        for k in key_to_delete:
            del S[k]

        # Calculate similarity set with the new region
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = calc_sim(R[t], R[n], img_size)

    regions = []
    for k, r in list(R.items()):
        regions.append({
            "rect": (
                r["min_x"],
                r["min_y"],
                r["max_x"] - r["min_x"],
                r["max_y"] - r["min_y"]
            ),
            "size": r["size"],
            "labels": r["labels"]
        })

    return img, regions

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

    times = []

    # for path_image in path_images:

    # img = skimage.io.imread('/home/mathi/Desktop/input_image.jpg')
    images_fil = glob.glob(path_images[7])
    images = [skimage.io.imread(img) for img in images_fil]
    # img = images[1]

    for i, img in enumerate(images):
        print(i)

        # Measure time
        tic = time.time()

        # Perform selective search
        img_lbl, regions = selective_search(
            src=img,
            scale=130,
            sigma=0.75,
            min_size=50
        )

        candidates = set()
        for r in regions:
            # Excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue

            # Excluding regions smaller than an specific number pixels
            if r['size'] < 1550:
                continue

            # Distorted rects
            x, y, w, h = r['rect']
            if w / h > 1.2 or h / w > 1.2:
                continue

            candidates.add(r['rect'])

        toc = time.time()
        times.append(toc - tic)

        # Draw rectangles on the original image and save it
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(img)
        ax.axis('off')
        for x, y, w, h in candidates:
            #print(x, y, w, h)
            rect = mpatches.Rectangle(
                xy=(x, y),
                width=w,
                height=h,
                fill=False,
                edgecolor='red',
                linewidth=1
            )
            ax.add_patch(rect)
        # path = str(Path('selective_search/test_images/ketchup/ketchup_'+str(i)+'.png').resolve())
        # plt.savefig(path)
        plt.show()
        plt.close()

    print('Average time performance:', (sum(times) / len(times)))

if __name__ == "__main__":
    main()
    plt.close('all')
