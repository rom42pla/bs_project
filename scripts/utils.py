from os import makedirs, remove
from os.path import join, exists, isfile
import urllib.request
from hashlib import md5
import tarfile
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image



def split_dataset(dataset, splits: list = [0.8, 0.1, 0.1], shuffles: list = [True, False, False], batch_size: int = 1):
    assert len(splits) >= 1 and len(shuffles) >= 1
    assert len(splits) == len(shuffles)
    for split, shuffle in zip(splits, shuffles):
        assert isinstance(split, float) and split > 0
        assert isinstance(shuffle, bool)
    assert np.isclose(np.sum(splits), 1)

    splits_amounts = []
    for i_split, split in enumerate(splits):
        if i_split < len(splits) - 1:
            splits_amounts += [int(split * len(dataset))]
        else:
            splits_amounts += [len(dataset) - np.sum(splits_amounts)]
    assert np.sum(splits_amounts) == len(dataset)

    subsets = random_split(dataset, splits_amounts)
    dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
                   for subset, shuffle in zip(subsets, shuffles)]
    return dataloaders


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

def read_json(filepath: str):
    with open(filepath, "r") as fp:
        return json.load(fp)


def save_json(d: dict, filepath: str):
    with open(filepath, "w") as fp:
        return json.dump(d, fp, indent=4)


def show_img(*imgs: torch.Tensor, filename: str = None, save_to_folder: str = None):
    assert not save_to_folder or isinstance(save_to_folder, str)
    imgs = list(imgs)
    for i_img, img in enumerate(imgs):
        assert isinstance(img, torch.Tensor)
        assert len(img.shape) == 3
        if save_to_folder:
            filename = filename if filename else f"img_{i_img}"
            save_image(img, join(save_to_folder, f"{filename}.png"))
        imgs[i_img] = img.permute(1, 2, 0).to("cpu").numpy()
    fig, axs = plt.subplots(1, len(imgs), squeeze=False)
    for i_ax, ax in enumerate(axs.flat):
        ax.imshow(imgs[i_ax])
    plt.show()


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return np.inf
    return 20 * torch.log10(1 / torch.sqrt(mse))
