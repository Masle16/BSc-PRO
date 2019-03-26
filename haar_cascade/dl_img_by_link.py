#!/mnt/sdb1/Anaconda/envs/BScPRO/bin/python

""" Module for downloading image by link """

###### IMPORTS ######
import os
import urllib.request
from pathlib import Path
import cv2
import numpy as np

###### GLOBAL VARIABLES ######
PATH = str(Path('domain_randomization').resolve())

###### FUNCTIONS ######
def store_raw_imgs():
    """ Store raw images """

    neg_img_link = 'image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513'
    neg_img_urls = urllib.request.urlopen(neg_img_link).read().decode()
    pic_num = 1

    # if not os.path.exists('neg'):
    #     os.makedirs('neg')

    for i in neg_img_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, PATH + '/neg/' + str(pic_num) + '.jpg')
            img = cv2.imread(PATH + '/neg/' + str(pic_num) + '.jpg', cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (224, 224))
            cv2.imwrite(PATH + '/neg/' + str(pic_num) + '.jpg', resized_img)
            pic_num += 1

        except Exception as error:
            print(str(error))

    print('DONE!')

    return 0

def find_uglies():
    """ Removes ugly images """

    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_img_path = str(file_type) + '/' + str(img)
                    ugly = cv2.imread('uglies/' + str(ugly))
                    question = cv2.imread(current_img_path)
                    if ugly.shape == question.shape and not np.bitwise_xor(ugly, question).any():
                        print(current_img_path)
                        os.remove(current_img_path)

                except Exception as error:
                    print(str(error))

    print('DONE!')

    return 0

def create_pos_n_neg():
    """ Creates the descriptor file """

    for file_type in [(PATH + '/neg')]:
        for img in os.listdir(file_type):
            if file_type == (PATH + '/pos'):
                line = file_type + '/' + img + ' 1 0 0 50 50\n'
                with open((PATH + '/info.dat'), 'a') as doc:
                    doc.write(line)
            elif file_type == (PATH + '/neg'):
                line = file_type + '/' + img + '\n'
                with open((PATH + '/bg.txt'), 'a') as doc:
                    doc.write(line)

    print('DONE!')

    return 0

###### MAIN FUNCTION ######
def main():
    """ Main function """

    #store_raw_imgs()
    #find_uglies()
    create_pos_n_neg()

    return 0

if __name__ == '__main__':
    main()
