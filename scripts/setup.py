from os.path import join
import time
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

from utils import load_lfw_dataset, split_dataset, show_img, read_json
from utils import psnr


class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        super(AddGaussianNoise, self).__init__()
        self.std, self.mean = std, mean

    def forward(self, X):
        return X + torch.randn(X.size()) * self.std + self.mean


# base class for each custom module
class CustomModule(nn.Module):
    def __init__(self, device: str = "auto"):
        super(CustomModule, self).__init__()
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} \
            else "cuda" if torch.cuda.is_available() else "cpu"


class Classifier(CustomModule):
    def __init__(self, num_classes: int = 13233, pretrained: bool = True,
                 device: str = "auto"):
        super(Classifier, self).__init__(device=device)

        assert isinstance(pretrained, bool)

        # takes the feature extractor layers of the model
        resnet = models.resnet18(pretrained=pretrained)
        for parameter in resnet.parameters():
            parameter.requires_grad = False

        # changes the last classification layer to tune the model for another task
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        for parameter in resnet.fc.parameters():
            parameter.requires_grad = True

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
          lr: float = 1e-4, epochs: int = 5, batches_per_epoch: int = None,
          data_augmentation_transforms=None,
          filepath: str = None, verbose: bool = True):
    # checks about model's parameters
    assert isinstance(model, nn.Module)
    assert isinstance(data_train, DataLoader)
    assert isinstance(data_val, DataLoader)
    assert not filepath or isinstance(filepath, str)
    # checks on other parameters
    assert isinstance(verbose, bool)
    assert isinstance(lr, float) and lr > 0
    assert isinstance(epochs, int) and epochs >= 1

    since = time.time()
    best_epoch_loss, best_model_weights = np.inf, \
                                          copy.deepcopy(model.state_dict())

    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    for epoch in range(epochs):
        for phase in ['train', 'val']:
            data = data_train if phase == "train" else data_val

            if phase == 'train':
                model.train()
            else:
                with torch.no_grad():
                    model.eval()

            batches_to_do = min(batches_per_epoch if batches_per_epoch else len(data), len(data))

            epoch_losses, epoch_f1, epoch_accuracy = np.zeros(shape=batches_to_do), \
                                                     np.zeros(shape=batches_to_do), \
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

                # applies some data augmentation
                if phase == "train" and data_augmentation_transforms:
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
                            "avg PSNR": np.mean(epoch_psnrs[:i_batch]),
                            "avg F1 scores": np.mean(epoch_f1[:i_batch]),
                            "avg accuracy": np.mean(epoch_accuracy[:i_batch]),
                            "time": "{:.0f}:{:.0f}".format(time_elapsed // 60, time_elapsed % 60)
                        }))

            # deep copy the model
            avg_epoch_loss = np.mean(epoch_losses)
            if phase == 'val' and avg_epoch_loss < best_epoch_loss:
                print(f"Found best model with loss {avg_epoch_loss}")
                best_epoch_loss, best_model_weights = avg_epoch_loss, \
                                                      copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_weights)
    # saves to a file
    if filepath:
        torch.save(model, filepath)
        print(f"Model saved to {filepath}")
    return model


if __name__ == "__main__":
    parameters_path = join("..", "parameters.json")
    assets_path = join("..", "assets")
    lfw_path = join(assets_path, "lfw")

    parameters = read_json(filepath=parameters_path)

    lfw_dataset_train, lfw_dataset_test = load_lfw_dataset(filepath=assets_path)
    labels = {label.item() for image, label in lfw_dataset_train}

    lfw_dataloader_train, lfw_dataloader_test = DataLoader(dataset=lfw_dataset_train, shuffle=True,
                                                           batch_size=parameters["training"]["batch_size"]), \
                                                DataLoader(dataset=lfw_dataset_test, shuffle=True,
                                                           batch_size=parameters["training"]["batch_size"])

    data_augmentation_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    classifier = Classifier(num_classes=len(labels), pretrained=True)
    train(classifier, epochs=parameters["training"]["epochs"],
          data_augmentation_transforms=data_augmentation_transforms,
          data_train=lfw_dataloader_train, data_val=lfw_dataloader_test)
