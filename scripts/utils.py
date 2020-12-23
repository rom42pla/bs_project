import os
from os import makedirs, remove
from os.path import join, exists, isfile
import urllib.request
from hashlib import md5
import tarfile

import numpy as np

import torchvision
from torchvision.datasets import ImageFolder

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, random_split
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader

def split_dataset(dataset, splits: list = [0.8, 0.1, 0.1]):
    assert len(splits) >= 1
    for split in splits:
        assert isinstance(split, float) and split > 0
    assert np.isclose(np.sum(splits), 1)

    splits_amounts = []
    for i_split, split in enumerate(splits):
        if i_split < len(splits)-1:
            splits_amounts += [int(split * len(dataset))]
        else:
            splits_amounts += [len(dataset) - np.sum(splits_amounts)]
    assert np.sum(splits_amounts) == len(dataset)

    subsets = torch.utils.data.random_split(dataset, splits_amounts)
    return subsets

def load_lfw_dataset(filepath: str, transform=None):
    # eventually creates empty directories
    if not exists(filepath):
        makedirs(filepath)
    lfw_path = join(filepath, "lfw")
    if not exists(lfw_path):
        # downloads the zipped file to the folder
        lfw_zipped_filepath = join(filepath, "lfw_zipped.tgz")
        if not exists(lfw_zipped_filepath):
            print(f"Downloading zipped file into {lfw_zipped_filepath}")
            urllib.request.urlretrieve("http://vis-www.cs.umass.edu/lfw/lfw.tgz", lfw_zipped_filepath)
        else:
            if md5(open(lfw_zipped_filepath, 'rb').read()).hexdigest() != "a17d05bd522c52d84eca14327a23d494":
                print(f"Zipped file {lfw_zipped_filepath} is corrupted and is now being re-downloaded")
                urllib.request.urlretrieve("http://vis-www.cs.umass.edu/lfw/lfw.tgz", lfw_zipped_filepath)
        # extracts the files
        print(f"Extracting the files")
        tarfile.open(lfw_zipped_filepath).extractall(filepath)
        # removes the zipped file
        remove(lfw_zipped_filepath)
    # loads the dataset as a tensor
    imagenet_data = ImageFolder(lfw_path, transform=transform)
    return imagenet_data
