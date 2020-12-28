from os import makedirs, remove
from os.path import join, exists, isfile
import urllib.request
from hashlib import md5
import tarfile
import json

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score

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
    X, y = np.transpose(lfw_dataset.images, (0, 3, 1, 2)) / 255, lfw_dataset.target

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
        imgs[i_img] = img.permute(1, 2, 0).detach().cpu().numpy()
    fig, axs = plt.subplots(1, len(imgs), squeeze=False)
    for i_ax, ax in enumerate(axs.flat):
        ax.imshow(imgs[i_ax])
    plt.show()


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return np.inf
    return 20 * torch.log10(1 / torch.sqrt(mse))


def plot_roc_curve(y, y_pred, labels, title: str = None):
    fpr, tpr, auc = [], [], []
    for i_class in labels:
        epoch_y_binary, epoch_y_pred_binary = [1 if label == i_class else 0 for label in y], \
                                              [1 if label == i_class else 0 for label in y_pred]
        class_fpr, class_tpr = np.zeros(len(epoch_y_binary)), np.zeros(len(epoch_y_binary))
        for i in range(len(epoch_y_binary)):
            tn, fp, fn, tp = confusion_matrix(epoch_y_binary[:i + 1], epoch_y_pred_binary[:i + 1],
                                              labels=[0, 1]).ravel()
            class_fpr[i], class_tpr[i] = (fp + 1) / (fp + tn + 1), \
                                         (tp + 1) / (tp + fn + 1)
        fpr += [class_fpr]
        tpr += [class_tpr]
        auc += [roc_auc_score(y_true=epoch_y_binary, y_score=epoch_y_pred_binary)]

    for i_label, label in enumerate(labels):
        plt.plot(fpr[i_label], tpr[i_label])
    if len(labels) <= 10:
        plt.legend([f"Label {label}, AUC {auc}" for label, auc in zip(labels, auc)])
    plt.title(f'ROC curve {"" if not title else title}')
    plt.tight_layout()
    plt.show()

def plot_losses(train_losses, test_losses, title: str = None):
    sns.lineplot(y=train_losses, x=range(1, len(train_losses)+1))
    sns.lineplot(y=test_losses, x=range(1, len(test_losses)+1))
    plt.title(f'Model loss {"" if not title else title}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

def plot_cmc(y, y_pred_scores, title: str = None):
    distance_matrix = np.asarray(y_pred_scores, dtype=np.float)
    cms = np.zeros(shape=distance_matrix.shape[0])
    for label, image_scores in zip(y, distance_matrix):
        ordered_indexes = np.flip(np.argsort(image_scores))
        rank = np.where(ordered_indexes == label)[0]
        cms[rank] += 1

    cms[0] = cms[0] / distance_matrix.shape[0]
    recognition_rate = cms[0]
    for k in range(1, distance_matrix.shape[0]):
        cms[k] = cms[k] / distance_matrix.shape[0] + cms[k - 1]

    cms_to_plot = 0
    for i in range(len(cms)):
        if np.allclose(cms[i:], cms[i]):
            cms_to_plot = i
            break

    sns.lineplot(y=cms[:cms_to_plot+2], x=range(1, len(cms[:cms_to_plot+2]) + 1))
    plt.title(f'CMC {"" if not title else title}')
    plt.ylabel('identification rate')
    plt.xlabel('rank')
    plt.show()

def plot_stats(accuracies, precisions, recalls, f1_scores, title: str = None):
    sns.lineplot(y=accuracies, x=range(1, len(accuracies)+1))
    sns.lineplot(y=precisions, x=range(1, len(accuracies)+1))
    sns.lineplot(y=recalls, x=range(1, len(accuracies)+1))
    sns.lineplot(y=f1_scores, x=range(1, len(accuracies)+1))
    plt.title(f'Stats {"" if not title else title}')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'precision', 'recall', 'F1 scores'], loc='lower right')
    plt.show()
