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

from models import FaceRecognitionModel
from utils import load_lfw_dataset, read_json
from utils import plot_roc_curve, plot_losses, plot_stats, psnr


def test(model: nn.Module, data: DataLoader,
         verbose: bool = True, resize: bool = False,
         plot_roc: bool = False, plot_loss: bool = True, plot_other_stats: bool = True):
    # checks about model's parameters
    assert isinstance(model, nn.Module)
    assert isinstance(data, DataLoader)
    # checks on other parameters
    assert isinstance(verbose, bool)
    assert isinstance(plot_loss, bool)
    assert isinstance(plot_roc, bool)
    assert isinstance(plot_other_stats, bool)

    since = time.time()
    total_y, total_y_pred, total_y_pred_scores = [], [], []
    labels = {label.item() for image, label in data.dataset}
    model.eval()

    for i_batch, batch in enumerate(data):
        # gets input data
        X, y = batch[0].to(model.device), \
               batch[1].to(model.device)

        # resizes the image
        if resize:
            square_edge = min(X.shape[2:])
            X_resized = torch.zeros(size=(X.shape[0], X.shape[1], square_edge, square_edge)).to(model.device)
            for i_img, img in enumerate(X):
                X_resized[i_img] = transforms.CenterCrop(square_edge)(img)
            X = X_resized

        # forward pass
        with torch.no_grad():
            y_pred = model(X)

        y_pred_labels = torch.argmax(y_pred, dim=-1)

        total_y += y.detach().cpu().tolist()
        total_y_pred += y_pred_labels.detach().cpu().tolist()
        total_y_pred_scores += [scores.detach().cpu().tolist() for scores in y_pred]

    '''
    E N D
    O F
    T E S T
    '''
    time_elapsed = time.time() - since

    print(pd.DataFrame(
        index=[model.name],
        data={
            "accuracy": accuracy_score(y_true=total_y, y_pred=total_y_pred),
            "precision": precision_score(y_true=total_y, y_pred=total_y_pred, average="macro"),
            "recall": recall_score(y_true=total_y, y_pred=total_y_pred, average="macro"),
            "f1 score": f1_score(y_true=total_y, y_pred=total_y_pred, average="macro"),
            "time": "{:.0f}:{:.0f}".format(time_elapsed // 60, time_elapsed % 60)
        }))

    distance_matrix = torch.as_tensor(total_y_pred_scores, dtype=torch.float)

    # if plot_roc:
    #     plot_roc_curve(y=total_y, y_pred=total_y_pred, labels=range(model.num_classes), title=model.name)


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

    lfw_dataloader_test = DataLoader(dataset=lfw_dataset_test, shuffle=True,
                                     batch_size=parameters["training"]["batch_size"])

    models = [
        FaceRecognitionModel(name="plain",
                             num_classes=len(labels),
                             add_noise=False,
                             do_denoising=False,
                             do_super_resolution=False,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        FaceRecognitionModel(name="n",
                             num_classes=len(labels),
                             add_noise=True,
                             do_denoising=False,
                             do_super_resolution=False,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        FaceRecognitionModel(name="n_dn",
                             num_classes=len(labels),
                             add_noise=True,
                             do_denoising=True,
                             do_super_resolution=False,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        FaceRecognitionModel(name="SR",
                             num_classes=len(labels),
                             add_noise=False,
                             do_denoising=False,
                             do_super_resolution=True,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        FaceRecognitionModel(name="n_SR",
                             num_classes=len(labels),
                             add_noise=True,
                             do_denoising=False,
                             do_super_resolution=True,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        FaceRecognitionModel(name="n_dn_SR",
                             num_classes=len(labels),
                             add_noise=True,
                             do_denoising=True, denoise_before_sr=True,
                             do_super_resolution=True,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        FaceRecognitionModel(name="n_SR_dn",
                             num_classes=len(labels),
                             add_noise=True,
                             do_denoising=True, denoise_before_sr=False,
                             do_super_resolution=True,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
    ]

    for model in models:
        try:
            model.load_state_dict(torch.load(join(models_path, f"frm_{model.name}.pth")))
        except:
            print(f"No weigths named frm_{model.name}.pth found")
            continue

        test(model, data=lfw_dataloader_test, resize=True,
             plot_loss=True, plot_roc=True, plot_other_stats=True)
