from os import makedirs, listdir
from os.path import join, exists
import json

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import auc

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
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


def load_flickr_faces_dataset(filepath: str):
    # eventually creates empty directories
    if not exists(filepath):
        raise Exception(f"{filepath} not found")

    flickr_dataset = []
    resizing_transforms = transforms.Compose([
        transforms.CenterCrop((125, 94)),
        transforms.ToTensor()
    ])
    for image_filename in listdir(filepath):
        image_filepath = join(filepath, image_filename)
        image = resizing_transforms(Image.open(image_filepath).convert("RGB"))
        flickr_dataset += [(image, torch.as_tensor(-1))]
    return flickr_dataset


def read_json(filepath: str):
    with open(filepath, "r") as fp:
        return json.load(fp)


def save_json(d: dict, filepath: str):
    with open(filepath, "w") as fp:
        return json.dump(d, fp, indent=4)


def show_img(*imgs: torch.Tensor, titles: list = None, plot_title: str = None,
             filename: str = None, save_to_folder: str = None):
    assert not save_to_folder or isinstance(save_to_folder, str)
    assert not titles or (titles and len(imgs) == len(titles))
    imgs = list(imgs)
    for i_img, img in enumerate(imgs):
        assert isinstance(img, torch.Tensor)
        assert len(img.shape) == 3
        if save_to_folder:
            filename = filename if filename else f"img_{i_img}"
            save_image(img, join(save_to_folder, f"{filename}.png"))
        imgs[i_img] = img.permute(1, 2, 0).detach().cpu().numpy()
    fig, axs = plt.subplots(1, len(imgs), squeeze=False)
    fig.set_size_inches(5 * len(imgs), 5)
    for i_ax, ax in enumerate(axs.flat):
        ax.imshow(imgs[i_ax])
        if titles:
            ax.set_title(titles[i_ax])
    if plot_title:
        fig.suptitle(plot_title)
    plt.tight_layout()
    plt.show()


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return np.inf
    return 20 * torch.log10(1 / torch.sqrt(mse))


def plot_labels_distribution(y):
    impostors, not_impostors = np.sum([1 for label in y if label == -1]), \
                               np.sum([1 for label in y if label != -1])

    fig1, ax1 = plt.subplots()
    ax1.pie((impostors, not_impostors), labels=["Impostors", "Not impostors"], explode=(0.1, 0),
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')

    plt.title("Impostors distribution")
    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, test_losses, title: str = None, filepath: str = None):
    # plots the loss chart
    sns.lineplot(y=train_losses, x=range(1, len(train_losses) + 1))
    sns.lineplot(y=test_losses, x=range(1, len(test_losses) + 1))
    plt.title(f'Loss {"" if not title else f"for model {title}"}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y, y_pred_scores, title: str = None, filepath: str = None):
    # data structures
    thresholds = np.asarray(np.linspace(start=0.01, stop=1, num=50, endpoint=True))
    distance_matrix = np.asarray(y_pred_scores, dtype=np.float)
    di, dir = np.zeros(shape=(len(thresholds), distance_matrix.shape[1])), \
              np.zeros(shape=(len(thresholds), distance_matrix.shape[1]))
    frr, far = np.zeros(shape=(len(thresholds))), np.zeros(shape=(len(thresholds)))
    tg, ti = np.sum([1 for label in y if label != -1]), \
             np.sum([1 for label in y if label == -1])
    # loops over each threshold
    for i_threshold, threshold in enumerate(thresholds):
        fa, gr = 0, 0
        # loops over each probe
        for label, image_scores in zip(y, distance_matrix):
            ordered_indexes, ordered_scores = np.flip(np.argsort(image_scores)), \
                                              np.flip(np.sort(image_scores))
            # if the system is sure this probe is in the gallery
            if ordered_scores[0] >= threshold:
                if label == ordered_indexes[0]:
                    di[i_threshold, 0] += 1
                else:
                    rank = np.where(ordered_indexes == label)[0]
                    if rank.size > 0:
                        di[i_threshold, rank] += 1
                    fa += 1
            else:
                gr += 1

        # rates
        dir[i_threshold, 0] = di[i_threshold, 0] / tg
        far[i_threshold], frr[i_threshold] = fa / ti, \
                                             1 - dir[i_threshold, 0]
        # updates DIR
        for k in range(1, distance_matrix.shape[0]):
            if di[i_threshold, k] == 0:
                break
            dir[i_threshold, k] = di[i_threshold, k] / tg + dir[i_threshold, k - 1]

    # plots the ROC curve
    sns.lineplot(y=dir[:, 0], x=far)
    plt.xlabel('FAR (False Acceptance Rate)')
    plt.ylabel('DIR (Detect and Identification Rate)')
    plt.title(f'Watchlist ROC {"" if not title else f"for model {title}"}\n'
              f'(AUC={np.round(auc(far, dir[:, 0]), 4)})')
    plt.tight_layout()
    if filepath:
        plt.savefig(join(filepath, f'roc.png'))
    plt.show()


    # plots the EER
    i_err = np.argmin(np.abs(far - frr))
    sns.lineplot(y=far, x=thresholds)
    sns.lineplot(y=frr, x=thresholds)
    plt.xlabel('thresholds')
    plt.ylabel('rate')
    plt.title(f'FAR and FRR {"" if not title else f"for model {title}"}\n'
              f'(EER={far[i_err]})')
    plt.annotate("EER", (thresholds[i_err], far[i_err] + 0.025))
    plt.plot(thresholds[i_err], far[i_err], "ro")
    plt.legend(["FAR (False Acceptance Rate)", "FRR (False Rejection Rate)"])
    plt.tight_layout()
    if filepath:
        plt.savefig(join(filepath, f'eer.png'))
    plt.show()


def plot_cmc(y, y_pred_scores,
             title: str = None, filepath: str = None):
    # data structures
    distance_matrix = np.asarray(y_pred_scores, dtype=np.float)
    cms = np.zeros(shape=distance_matrix.shape[0])
    # loops over each probe
    for label, image_scores in zip(y, distance_matrix):
        ordered_indexes = np.flip(np.argsort(image_scores))
        rank = np.where(ordered_indexes == label)[0]
        cms[rank] += 1

    cms[0] = cms[0] / distance_matrix.shape[0]
    for k in range(1, distance_matrix.shape[0]):
        cms[k] = cms[k] / distance_matrix.shape[0] + cms[k - 1]

    # trims unuseful parts of the plot
    cms_to_plot = 0
    for i in range(len(cms)):
        if np.allclose(cms[i:], cms[i]):
            cms_to_plot = i
            break

    # plots the CMC
    sns.lineplot(y=cms[:cms_to_plot + 2], x=range(1, len(cms[:cms_to_plot + 2]) + 1))
    plt.title(f'CMC {"" if not title else f"for model {title}"}\n'
              f'(RR={np.round(cms[0], 4)}, CMS(5)={np.round(cms[5], 4)}, CMS(10)={np.round(cms[10], 4)})')
    plt.ylabel('identification rate')
    plt.xlabel('rank')
    plt.tight_layout()
    if filepath:
        plt.savefig(join(filepath, f'cmc.png'))
    plt.show()
