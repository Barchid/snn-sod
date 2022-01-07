import os
import random
import cv2
import imutils
import torch
import numpy as np
from project.datamodules.oxford_iiit_pet import OxfordPetDatamodule, preprocess_mask
import matplotlib.pyplot as plt

POLYGON_POINTS = 19  # polygon annotation of 11 vertices


def find_polygon(mask: np.ndarray):
    # get contour from mask
    cnts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)

    for i in range(1, 1000):
        eps = i * 0.0001
        approx = cv2.approxPolyDP(c, eps * peri, True)

        if len(approx) == POLYGON_POINTS:
            print('FOUND ! After ', i, 'iterations')
            return approx
        if len(approx) < POLYGON_POINTS:
            print('Not found exact match... Approximation length :', len(approx))
            return approx


if __name__ == "__main__":

    dm = OxfordPetDatamodule(1, data_dir="/home/sami/codes/snn-decolle-segmentation/data/oxford_iiit_pet")
    dm.setup()

    train_set = dm.train_set

    # get item
    for i in range(len(train_set.images_filenames)):
        image_filename = train_set.images_filenames[i]
        image = cv2.imread(os.path.join(train_set.images_directory, image_filename))

        mask = cv2.imread(
            os.path.join(train_set.masks_directory,
                         image_filename.replace(".jpg", ".png")),
            cv2.IMREAD_UNCHANGED,
        )

        mask = preprocess_mask(mask)
        contour = find_polygon(mask)

        cv2.drawContours(mask, [contour], -1, 1., 3)
        plt.imshow(mask)
        plt.show()
        height, width = mask.shape[0], mask.shape[1]
