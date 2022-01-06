from collections import defaultdict
import copy
import random
import os
import shutil
from typing import Optional
from urllib.request import urlretrieve
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# get the class from the name of image files
class_dic = {
    "Abyssinian": 0,
    "american_bulldog": 1,
    "american_pit_bull_terrier": 1,
    "basset_hound": 1,
    "beagle": 1,
    "Bengal": 0,
    "Birman": 0,
    "Bombay": 0,
    "boxer": 1,
    "British_Shorthair": 0,
    "chihuahua": 1,
    "Egyptian_Mau": 0,
    "english_cocker_spaniel": 1,
    "english_setter": 1,
    "german_shorthaired": 1,
    "great_pyrenees": 1,
    "havanese": 1,
    "japanese_chin": 1,
    "keeshond": 1,
    "leonberger": 1,
    "Maine_Coon": 0,
    "miniature_pinscher": 1,
    "newfoundland": 1,
    "Persian": 0,
    "pomeranian": 1,
    "pug": 1,
    "Ragdoll": 0,
    "Russian_Blue": 0,
    "saint_bernard": 1,
    "samoyed": 1,
    "scottish_terrier": 1,
    "shiba_inu": 1,
    "Siamese": 0,
    "Sphynx": 0,
    "staffordshire_bull_terrier": 1,
    "wheaten_terrier": 1,
    "yorkshire_terrier": 1,
}

################################################################################################################
# Function to download & unzip Oxford-IIT-Pet dataset
################################################################################################################


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        print("Dataset already exists on the disk. Skipping download.")
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    shutil.unpack_archive(filepath, extract_dir)


class OxfordPetDatamodule(pl.LightningDataModule):
    TRAIN_COUNT = 6000  # 10000
    VAL_COUNT = 1300  # 2091

    def __init__(self, batch_size: int, data_dir="data/oxford_iiit_pet", is_grayscale=False, **kwargs):
        # create the directory if not exist
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.is_grayscale = is_grayscale

    def prepare_data(self) -> None:
        # Download & extract dataset
        ################################################################################################################
        if os.path.isdir(os.path.join(self.data_dir, "images")) and os.path.isdir(
            os.path.join(self.data_dir, "images")
        ):
            print("OXFORD IIT dataset already exists. Skip download & extract")
        else:
            # images
            filepath = os.path.join(self.data_dir, "images.tar.gz")
            download_url(
                url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
                filepath=filepath,
            )
            extract_archive(filepath)

            # segmentation masks
            filepath = os.path.join(self.data_dir, "annotations.tar.gz")
            download_url(
                url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
                filepath=filepath,
            )
            extract_archive(filepath)

    def setup(self, stage: Optional[str] = None) -> None:
        ################################################################################################################
        # Split train and test sets
        ################################################################################################################
        root_directory = os.path.join(self.data_dir)
        images_directory = os.path.join(root_directory, "images")
        masks_directory = os.path.join(root_directory, "annotations", "trimaps")
        xml_directory = os.path.join(root_directory, "annotations", "xmls")

        images_filenames = list(sorted(os.listdir(images_directory)))
        correct_images_filenames = [
            i
            for i in images_filenames
            if cv2.imread(os.path.join(images_directory, i)) is not None
        ]

        random.shuffle(correct_images_filenames)

        train_images_filenames = correct_images_filenames[:self.TRAIN_COUNT]
        val_images_filenames = correct_images_filenames[
            self.TRAIN_COUNT: (self.TRAIN_COUNT + self.VAL_COUNT)
        ]

        self.train_set = OxfordPetDataset(
            train_images_filenames,
            images_directory,
            masks_directory,
            transform=get_transforms(224, 224, is_training=True, is_grayscale=self.is_grayscale),
            use_DOG=False,
            is_grayscale=self.is_grayscale
        )

        self.val_set = OxfordPetDataset(
            val_images_filenames,
            images_directory,
            masks_directory,
            transform=get_transforms(224, 224, is_training=False, is_grayscale=self.is_grayscale),
            use_DOG=False,
            is_grayscale=self.is_grayscale,
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)


################################################################################################################
# Utils functions
################################################################################################################
# Create a binary mask


def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask


def DOG_transform(image, sigma1=1.0, sigma2=4.0, kernel_size=7):
    g1 = cv2.GaussianBlur(image, (kernel_size, kernel_size),
                          sigmaX=sigma1, sigmaY=sigma1)
    g2 = cv2.GaussianBlur(image, (kernel_size, kernel_size),
                          sigmaX=sigma2, sigmaY=sigma2)

    DOG = g1 - g2

    return DOG


def on_off_filtering(image, sigma_center=1.0, sigma_surround=4.0, kernel_size=7):
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.

    G_center = cv2.GaussianBlur(image, (kernel_size, kernel_size),
                                sigmaX=sigma_center, sigmaY=sigma_center)
    G_surround = cv2.GaussianBlur(image, (kernel_size, kernel_size),
                                  sigmaX=sigma_surround, sigmaY=sigma_surround)
    DOG = G_center - G_surround

    ON = np.clip(DOG, 0., None)
    OFF = np.clip(-DOG, 0., None)
    on_off = np.stack([ON, OFF], axis=2)  # .astype(np.float32)
    # ensure range between 0 and 1
    on_off /= np.max(on_off)
    on_off *= 255
    return on_off.astype(np.uint8)


def get_transforms(height, width, is_training=False, is_grayscale=True):
    if is_training:
        return A.Compose(
            [
                A.Resize(height, width),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=(0.) if is_grayscale else (0.485, 0.456, 0.406),
                    std=(1.) if is_grayscale else (0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
        transf = [
            A.Resize(height, width),
            A.Normalize(
                mean=(0.) if is_grayscale else (0.485, 0.456, 0.406),
                std=(1.) if is_grayscale else (0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
        return A.Compose(
            transf
        )

################################################################################################################
# Oxfor dataset
################################################################################################################


class OxfordPetDataset(Dataset):
    def __init__(
        self, images_filenames, images_directory, masks_directory, transform=None, use_DOG=False, is_grayscale=True
    ):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform
        self.use_DOG = use_DOG
        self.is_grayscale = is_grayscale

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        if self.is_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, 2)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_DOG:
            image = on_off_filtering(image)

        # get class from classes directory
        class_id = get_class_from_filename(image_filename)

        mask = cv2.imread(
            os.path.join(self.masks_directory,
                         image_filename.replace(".jpg", ".png")),
            cv2.IMREAD_UNCHANGED,
        )

        mask = preprocess_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # get bounding box from mask
        # x_min, y_min, x_max, y_max = get_bbox_from_mask(mask.numpy())

        return image, class_id, mask  # torch.Tensor([x_min, y_min, x_max, y_max])


def get_class_from_filename(filename):
    filename = filename.replace('_AUGMENTED', '')
    breed_name = filename[
        : filename.rfind("_")
    ]  # get the name of the breed by removing the useless part "_{number}.png"

    return class_dic[breed_name]  # use the class dictionary (0=cat and 1=dog)
