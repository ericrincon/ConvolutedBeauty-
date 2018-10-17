import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from convolutedbeauty.data.utils import get_images


class BeautyDataSet(Dataset):
    """Dataset for the training of (dubious) "beauty" models using (admittedly low quality) images"""

    def __init__(self, dir_path, stratify_gender=False, transform=None):
        self.dir_path = dir_path
        self.stratify_gender = stratify_gender
        self.image_files = self._filter_files(get_images(dir_path))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def _get_attrs(self, filepath):
        split_name = filepath.split("_")

        gender, username, rating = None, None, None

        if len(split_name) >= 3:
            gender, rating = split_name[0], split_name[-1]

            gender = gender.split('/')[-1]
            username = "_".join(x for x in split_name[1:len(split_name) - 2])
            rating = float(rating.split(".")[0])

        return gender, username, rating

    def _filter_files(self, files):
        filtered_files = []

        for file in files:
            attrs = self._get_attrs(file)

            if not any(map(lambda x: x is None, attrs)):
                filtered_files.append([file] + list(attrs))

        return filtered_files

    def __getitem__(self, idx):
        file_path, gender, username, rating = self.image_files[idx]
        image = io.imread(file_path)

        sample = {'image': image, 'gender': gender, "rating": rating}

        if self.transform:
            sample = self.transform(sample)

        return sample
