from os import makedirs, remove
from os.path import join, exists, isfile
import urllib.request
from hashlib import md5
import tarfile
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people

import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image


def split_dataset(dataset: Dataset, splits: list, shuffles: list = None,
                  batch_size: int = 1):
    if shuffles is None:
        shuffles = [False for _ in splits]
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


def load_lfw_dataset(filepath: str, min_faces_per_person: int = 20):
    # eventually creates empty directories
    if not exists(filepath):
        makedirs(filepath)
    # downloads or fetches the dataset
    lfw_dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=1, color=True,
                                   funneled=True, data_home=filepath)
    X, y = np.transpose(lfw_dataset.images, (0, 3, 1, 2))/255, lfw_dataset.target

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=42)
    # zips Xs and ys
    lfw_train_dataset, lfw_test_dataset = list(zip(torch.from_numpy(X_train), torch.from_numpy(y_train))), \
                                          list(zip(torch.from_numpy(X_test), torch.from_numpy(y_test)))
    return lfw_train_dataset, lfw_test_dataset


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
