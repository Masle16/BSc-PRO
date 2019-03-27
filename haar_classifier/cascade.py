#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

""" Module for cascade classification """

######  IMPORTS ######
import glob
from pathlib import Path
import cv2
from background_subtration import background_sub, run_avg

###### GLOBAL VARIABLES ######

###### FUNCTIONS ######

###### MAIN FUNCTION ######
def main():
    """ Main function """

    ################## IMPORT IMAGES ##################

    # Baggrund
    path = str(Path('images_1280x720/baggrund/bevægelse/*.jpg').resolve())
    background_fil = glob.glob(path)
    background_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in background_fil]

    # Guleroedder
    path = str(Path('images_1280x720/gulerod/still/*.jpg'))
    carrot_fil = glob.glob(path)
    carrot_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in carrot_fil]

    # Kartofler
    path = str(Path('images_1280x720/kartofler/still/*.jpg').resolve())
    potato_fil = glob.glob(path)
    potato_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in potato_fil]

    # Kat laks
    path = str(Path('images_1280x720/kat_laks/still/*.jpg').resolve())
    cat_sal_fil = glob.glob(path)
    cat_sal_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_sal_fil]

    # Kat okse
    path = str(Path('images_1280x720/kat_okse/still/*.jpg').resolve())
    cat_beef_fil = glob.glob(path)
    cat_beef_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in cat_beef_fil]

    path = str(Path('preprocessing/background_mask.jpg').resolve())
    background_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    background_img = run_avg(background_images)

    ################## DETECT OBJECT ##################

    # Create cascade classifier
    path = str(Path('haar_classifier/cascade.xml').resolve())
    potato_cascade = cv2.CascadeClassifier(path)

    for img in potato_images:
        # Get region of interest
        roi, _ = background_sub(img, background_img, background_mask)
        (x_left, x_right, y_up, y_down) = roi

        dst = img.copy()
        cv2.rectangle(img=dst,
                      pt1=(x_left, y_up),
                      pt2=(x_right, y_down),
                      color=(255, 0, 0),
                      thickness=4)

        roi = img[y_up : y_down, x_left : x_right]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Detect object
        potatoes = potato_cascade.detectMultiScale(roi, 50, 50)

        for (x, y, width, height) in potatoes:
            cv2.rectangle(img=dst,
                          pt1=(x, y),
                          pt2=(x + width, y + height),
                          color=(0, 0, 255),
                          thickness=4)

            cv2.putText(img=dst,
                        text='Potato',
                        org=(x, y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA)

        cv2.imshow('Image', dst)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
