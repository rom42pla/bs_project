from os.path import join
import time
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import FaceRecognitionModel
from utils import load_lfw_dataset, read_json
from utils import plot_roc_curve, plot_losses, psnr, show_img


def train(model: nn.Module, data_train: DataLoader, data_val: DataLoader,
          lr: float = 1e-5, epochs: int = 5, batches_per_epoch: int = None,
          data_augmentation_transforms=None, resize: bool = False,
          filepath: str = None, verbose: bool = True,
          plot_roc: bool = False, plot_loss: bool = True, plot_other_stats: bool = True):
    # checks about model's parameters
    assert isinstance(model, nn.Module)
    assert isinstance(data_train, DataLoader)
    assert isinstance(data_val, DataLoader)
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

    scaler = torch.cuda.amp.GradScaler()

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

            for i_batch, batch in enumerate(data):
                # eventually early stops the training
                if batches_per_epoch and i_batch >= batches_to_do:
                    break

                # gets input data
                X, y = batch[0].to(model.device), \
                       batch[1].to(model.device)

                optimizer.zero_grad()

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
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(phase == 'train'):
                        y_pred = model(X)

                    ce_loss = nn.CrossEntropyLoss()(y_pred, y)
                    loss = ce_loss

                # backward pass
                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                y_pred_labels = torch.argmax(y_pred.detach().cpu(), dim=-1)

                epoch_y += y.detach().cpu().tolist()
                epoch_y_pred += y_pred_labels.detach().cpu().tolist()

                epoch_losses[i_batch] = loss

                # statistics
                if verbose and i_batch in np.linspace(start=1, stop=batches_to_do, num=1, dtype=np.int):
                    time_elapsed = time.time() - since
                    print(pd.DataFrame(
                        index=[
                            f"batch {i_batch + 1} of {batches_to_do}"],
                        data={
                            "model name": model.name,
                            "epoch": epoch,
                            "phase": phase,
                            "avg loss": np.mean(epoch_losses[:i_batch]),
                            # "avg PSNR": np.mean(epoch_psnrs[:i_batch]),
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
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    # if plot_roc:
    #     plot_roc_curve(y=epoch_y, y_pred=epoch_y_pred, labels=range(model.num_classes), title=model.name)
    # if plot_other_stats:
    #     plot_stats(accuracies=accuracies, precisions=precisions, recalls=recalls, f1_scores=f1_scores, title=model.name)
    if plot_loss:
        plot_losses(train_losses=train_losses, test_losses=test_losses, title=model.name)

    return model


if __name__ == "__main__":
    parameters_path = join("..", "parameters.json")
    parameters = read_json(filepath=parameters_path)

    # setting the seeds for reproducibility
    seed = parameters["training"]["seed"]
    torch.manual_seed(seed=seed)
    np.random.seed(seed=seed)

    assets_path = join("..", "assets")

    lfw_path = join(assets_path, "lfw")

    models_path = join(assets_path, "models")
    face_recognition_model_weights_path = join(models_path, "face_recognition_model_weights.pth")
    rrdb_pretrained_weights_path = join(models_path, "RRDB_PSNR_x4.pth")

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

    models = [
        FaceRecognitionModel(name="plain",
                             num_classes=len(labels),
                             add_noise=False,
                             do_denoising=False,
                             do_super_resolution=False,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        # FaceRecognitionModel(name="n",
        #                      num_classes=len(labels),
        #                      add_noise=True,
        #                      do_denoising=False,
        #                      do_super_resolution=False,
        #                      noise_prob=parameters["training"]["noise_prob"],
        #                      resnet_pretrained=True,
        #                      rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        # FaceRecognitionModel(name="n_dn",
        #                      num_classes=len(labels),
        #                      add_noise=True,
        #                      do_denoising=True,
        #                      do_super_resolution=False,
        #                      noise_prob=parameters["training"]["noise_prob"],
        #                      resnet_pretrained=True,
        #                      rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        # FaceRecognitionModel(name="SR",
        #                      num_classes=len(labels),
        #                      add_noise=False,
        #                      do_denoising=False,
        #                      do_super_resolution=True,
        #                      noise_prob=parameters["training"]["noise_prob"],
        #                      resnet_pretrained=True,
        #                      rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        # FaceRecognitionModel(name="n_SR",
        #                      num_classes=len(labels),
        #                      add_noise=True,
        #                      do_denoising=False,
        #                      do_super_resolution=True,
        #                      noise_prob=parameters["training"]["noise_prob"],
        #                      resnet_pretrained=True,
        #                      rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        # FaceRecognitionModel(name="n_dn_SR",
        #                      num_classes=len(labels),
        #                      add_noise=True,
        #                      do_denoising=True, denoise_before_sr=True,
        #                      do_super_resolution=True,
        #                      noise_prob=parameters["training"]["noise_prob"],
        #                      resnet_pretrained=True,
        #                      rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        # FaceRecognitionModel(name="n_SR_dn",
        #                      num_classes=len(labels),
        #                      add_noise=True,
        #                      do_denoising=True, denoise_before_sr=False,
        #                      do_super_resolution=True,
        #                      noise_prob=parameters["training"]["noise_prob"],
        #                      resnet_pretrained=True,
        #                      rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
    ]

    for model in models:
        train(model, epochs=parameters["training"]["epochs"],
              data_augmentation_transforms=data_augmentation_transforms, resize=True,
              data_train=lfw_dataloader_train, data_val=lfw_dataloader_test,
              plot_loss=True, plot_roc=True, plot_other_stats=True,
              filepath=join(models_path, f"frm_{model.name}.pth"))
