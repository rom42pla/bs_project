from os.path import join
import time
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

from utils import load_lfw_dataset, split_dataset, show_img, read_json
from utils import plot_roc_curve, plot_losses, plot_stats, psnr


# base class for each custom module
class CustomModule(nn.Module):
    def __init__(self, device: str = "auto"):
        super(CustomModule, self).__init__()
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} \
            else "cuda" if torch.cuda.is_available() else "cpu"


class SaltAndPepperNoise(CustomModule):
    def __init__(self, prob: float = 0.005,
                 device: str = "auto"):
        super(SaltAndPepperNoise, self).__init__(device=device)
        assert isinstance(prob, float) and 0 <= prob <= 1
        self.prob = prob

        self.to(self.device)

    def forward(self, X):
        mask = torch.rand(size=X.shape).to(self.device) >= self.prob
        return X * mask
        # return X + torch.randn(X.size()).to(self.device) * self.std + self.mean


class Classifier(CustomModule):
    def __init__(self, num_classes: int = 13233, pretrained: bool = True,
                 device: str = "auto"):
        super(Classifier, self).__init__(device=device)

        assert isinstance(pretrained, bool)
        assert isinstance(num_classes, int) and num_classes >= 2
        self.num_classes = num_classes

        # takes the feature extractor layers of the model
        resnet = models.resnet18(pretrained=pretrained)
        # changes the last classification layer to tune the model for another task
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

        self.layers = nn.Sequential(
            resnet
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        X = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(X)
        out = self.layers(X)

        if not self.training:
            out = F.log_softmax(out, dim=-1)
        return out


def train(model: nn.Module, data_train: DataLoader, data_val: DataLoader,
          lr: float = 1e-5, epochs: int = 5, batches_per_epoch: int = None,
          data_augmentation_transforms=None, resize: bool = False, add_noise: bool = True,
          filepath: str = None, verbose: bool = True,
          plot_roc: bool = False, plot_loss: bool = True, plot_other_stats: bool = True):
    # checks about model's parameters
    assert isinstance(model, nn.Module)
    assert isinstance(data_train, DataLoader)
    assert isinstance(data_val, DataLoader)
    assert isinstance(add_noise, bool)
    assert not filepath or isinstance(filepath, str)
    # checks on other parameters
    assert isinstance(verbose, bool)
    assert isinstance(plot_loss, bool)
    assert isinstance(plot_roc, bool)
    assert isinstance(plot_other_stats, bool)
    assert isinstance(lr, float) and lr > 0
    assert isinstance(epochs, int) and epochs >= 1

    since = time.time()
    best_epoch_loss, best_model_weights = np.inf, \
                                          copy.deepcopy(model.state_dict())
    train_losses, test_losses = [], []
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    for epoch in range(epochs):
        for phase in ['train', 'val']:
            '''
            S T A R T
            O F
            E P O C H
            '''

            data = data_train if phase == "train" else data_val

            if phase == 'train':
                model.train()
            else:
                with torch.no_grad():
                    model.eval()

            batches_to_do = min(batches_per_epoch if batches_per_epoch else len(data), len(data))

            epoch_losses = np.zeros(shape=batches_to_do)
            epoch_y, epoch_y_pred = [], []
            epoch_f1, epoch_accuracy = np.zeros(shape=batches_to_do), \
                                       np.zeros(shape=batches_to_do)
            epoch_ce_losses, epoch_psnrs = np.zeros(shape=batches_to_do), \
                                           np.zeros(shape=batches_to_do)

            for i_batch, batch in enumerate(data):
                # eventually early stops the training
                if batches_per_epoch and i_batch >= batches_to_do:
                    break

                # gets input data
                X, y = batch[0].to(model.device), \
                       batch[1].to(model.device)

                if add_noise:
                    X = SaltAndPepperNoise(device=model.device)(X)

                # resizes the image
                if resize:
                    square_edge = min(X.shape[2:])
                    X_resized = torch.zeros(size=(X.shape[0], X.shape[1], square_edge, square_edge)).to(model.device)
                    for i_img, img in enumerate(X):
                        X_resized[i_img] = transforms.RandomCrop(square_edge)(img) if phase == "train" \
                            else transforms.CenterCrop(square_edge)(img)
                    X = X_resized

                # applies some data augmentation
                if data_augmentation_transforms:
                    for i_img, img in enumerate(X):
                        X[i_img] = data_augmentation_transforms(img)

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(X)

                ce_loss = nn.CrossEntropyLoss()(y_pred, y)
                loss = ce_loss

                # backward pass
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                y_pred_labels = torch.argmax(y_pred.detach().cpu(), dim=-1)

                epoch_y += y.detach().cpu().tolist()
                epoch_y_pred += y_pred_labels.detach().cpu().tolist()

                epoch_losses[i_batch], epoch_f1[i_batch], epoch_accuracy[i_batch] = ce_loss, \
                                                                                    f1_score(
                                                                                        y_true=y.detach().cpu().numpy(),
                                                                                        y_pred=y_pred_labels.cpu().numpy(),
                                                                                        average="macro"), \
                                                                                    accuracy_score(
                                                                                        y_true=y.detach().cpu().numpy(),
                                                                                        y_pred=y_pred_labels.cpu().numpy()),
                epoch_ce_losses[i_batch], epoch_psnrs[i_batch], = ce_loss, psnr(X, X)

                # statistics
                if verbose and i_batch in np.linspace(start=1, stop=batches_to_do, num=5, dtype=np.int):
                    time_elapsed = time.time() - since
                    print(pd.DataFrame(
                        index=[
                            f"batch {i_batch + 1} of {batches_to_do}"],
                        data={
                            "epoch": epoch,
                            "phase": phase,
                            "avg loss": np.mean(epoch_losses[:i_batch]),
                            # "avg PSNR": np.mean(epoch_psnrs[:i_batch]),
                            "avg F1": np.mean(epoch_f1[:i_batch]),
                            "avg accuracy": np.mean(epoch_accuracy[:i_batch]),
                            "time": "{:.0f}:{:.0f}".format(time_elapsed // 60, time_elapsed % 60)
                        }))

            '''
            E N D
            O F
            E P O C H
            '''
            # deep copy the model
            avg_epoch_loss = np.mean(epoch_losses)
            if phase == 'val' and avg_epoch_loss < best_epoch_loss:
                print(f"Found best model with loss {avg_epoch_loss}")
                best_epoch_loss, best_model_weights = avg_epoch_loss, \
                                                      copy.deepcopy(model.state_dict())
            if phase == "train":
                train_losses += [avg_epoch_loss]
            else:
                test_losses += [avg_epoch_loss]
                accuracies += [accuracy_score(y_true=epoch_y, y_pred=epoch_y_pred)]
                precisions += [precision_score(y_true=epoch_y, y_pred=epoch_y_pred, average="macro")]
                recalls += [recall_score(y_true=epoch_y, y_pred=epoch_y_pred, average="macro")]
                f1_scores += [f1_score(y_true=epoch_y, y_pred=epoch_y_pred, average="macro")]

    '''
    E N D
    O F
    T R A I N I N G
    '''
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_weights)
    # saves to a file
    if filepath:
        torch.save(model, filepath)
        print(f"Model saved to {filepath}")

    if plot_roc:
        plot_roc_curve(y=epoch_y, y_pred=epoch_y_pred, labels=range(model.num_classes))
    if plot_other_stats:
        plot_stats(accuracies=accuracies, precisions=precisions, recalls=recalls, f1_scores=f1_scores)
    if plot_loss:
        plot_losses(train_losses=train_losses, test_losses=test_losses)

    return model


if __name__ == "__main__":
    parameters_path = join("..", "parameters.json")
    assets_path = join("..", "assets")
    lfw_path = join(assets_path, "lfw")

    parameters = read_json(filepath=parameters_path)

    lfw_dataset_train, lfw_dataset_test = load_lfw_dataset(filepath=assets_path,
                                                           min_faces_per_person=parameters["data"][
                                                               "min_faces_per_person"])
    labels = {label.item() for image, label in lfw_dataset_train}

    lfw_dataloader_train, lfw_dataloader_test = DataLoader(dataset=lfw_dataset_train, shuffle=True,
                                                           batch_size=parameters["training"]["batch_size"]), \
                                                DataLoader(dataset=lfw_dataset_test, shuffle=True,
                                                           batch_size=parameters["training"]["batch_size"])

    data_augmentation_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, saturation=0.1, contrast=0.1, hue=0.1),
        transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
        transforms.RandomAffine(degrees=15),
        transforms.ToTensor()
    ])

    classifier = Classifier(num_classes=len(labels), pretrained=True)
    train(classifier, epochs=parameters["training"]["epochs"],
          data_augmentation_transforms=data_augmentation_transforms, resize=True, add_noise=True,
          data_train=lfw_dataloader_train, data_val=lfw_dataloader_test,
          plot_loss=True, plot_roc=True, plot_other_stats=True)
