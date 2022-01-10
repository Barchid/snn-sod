import os
import random
import cv2
import imutils
import torch
import numpy as np
from project.datamodules.oxford_iiit_pet import OxfordPetDatamodule, preprocess_mask
import matplotlib.pyplot as plt
import argparse
import json


def is_clockwise(contour):
    sum = 0.
    pt1 = 0
    pt2 = 1
    for _ in range(len(contour) - 1):
        x1, y1 = contour[pt1]
        x2, y2 = contour[pt2]

        sum += (x2 - x1) * (y2 + y1)

        pt1 += 1
        pt2 += 1

    # link last point to first point
    x1, y1 = contour[-1]
    x2, y2 = contour[0]
    sum += (x2 - x1) * (y2 + y1)

    return sum >= 0


def find_polygon(mask: np.ndarray, polygon_vertices):
    # get contour from mask
    cnts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        print('no contours in the mask.')
        return None

    c = max(cnts, key=cv2.contourArea)

    # return cv2.convexHull(c, clockwise=False)
    peri = cv2.arcLength(c, True)

    for i in range(1, 1000):
        eps = i * 0.0001
        approx = cv2.approxPolyDP(c, eps * peri, True)

        if len(approx) == polygon_vertices:
            # print('FOUND ! After ', i, 'iterations')
            return approx
        if len(approx) < polygon_vertices:
            # print('Not found exact match... Approximation length :', len(approx))

            # if the approximate is shorter than polygon_vertices, adds the same last point
            missed = polygon_vertices - len(approx)
            for _ in range(missed):
                approx = np.append(approx, np.expand_dims(approx[-1], axis=0), axis=0)

            return approx


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data/oxford_iiit_pet")
    parser.add_argument('--polygon_vertices', type=int, default=19)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    data_dir = args.data_dir #"/home/sami/codes/snn-decolle-segmentation/data/oxford_iiit_pet"
    images_directory = os.path.join(data_dir, "images")
    masks_directory = os.path.join(data_dir, "annotations", "trimaps")

    # key-value of
    annotation_dict = {}

    images_filenames = list(sorted(os.listdir(images_directory)))
    correct_images_filenames = [
        i
        for i in images_filenames
        if cv2.imread(os.path.join(images_directory, i)) is not None
    ]

    # get item
    for i in range(len(correct_images_filenames)):
        image_filename = correct_images_filenames[i]
        image = cv2.imread(os.path.join(images_directory, image_filename))

        mask = cv2.imread(
            os.path.join(masks_directory,
                         image_filename.replace(".jpg", ".png")),
            cv2.IMREAD_UNCHANGED,
        )

        mask = preprocess_mask(mask)

        contour = find_polygon(mask, args.polygon_vertices)

        if contour is None:
            continue


        # cv2.drawContours(mask, [contour], -1, 1., cv2.FILLED)
        # plt.imshow(mask)
        # plt.show()
        # exit()
        
        contour = np.squeeze(contour).tolist()
        annotation_dict[image_filename] = contour

        # height, width = mask.shape[0], mask.shape[1]

    with open(f'polygons_vertices{args.polygon_vertices}.json', 'w') as file:
        json.dump(annotation_dict, file)

    print('done !')
