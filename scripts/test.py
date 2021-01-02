from os import makedirs
from os.path import join
import time

import numpy as np
import pandas as pd

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from models import FaceRecognitionModel
import utils
from utils import load_lfw_dataset, load_flickr_faces_dataset, read_json


def test(model: nn.Module, data: DataLoader,
         resize: bool = False,
         plot_roc: bool = False, plot_cmc: bool = True, test_folder: str = None):
    # checks about model's parameters
    assert isinstance(model, nn.Module)
    assert isinstance(data, DataLoader)
    # checks on other parameters
    assert isinstance(resize, bool)
    assert isinstance(plot_roc, bool)
    assert isinstance(plot_cmc, bool)

    since = time.time()
    total_y_open_set, total_y_pred_scores_open_set = [], []
    total_y_closed_set, total_y_pred_scores_closed_set = [], []
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

        # wraps the results for the open set part of the task
        total_y_open_set += [y[i].item()
                             for i, label in enumerate(y)]
        total_y_pred_scores_open_set += [y_pred[i].detach().cpu().tolist()
                                         for i, label in enumerate(y)]
        # wraps the results for the closed set part of the task
        total_y_closed_set += [y[i].item()
                               for i, label in enumerate(y) if label.item() != -1]
        total_y_pred_scores_closed_set += [y_pred[i].detach().cpu().tolist()
                                           for i, label in enumerate(y) if label.item() != -1]

    '''
    E N D
    O F
    T E S T
    '''
    time_elapsed = time.time() - since

    print(pd.DataFrame(
        index=[model.name],
        data={
            "time": "{:.0f}:{:.0f}".format(time_elapsed // 60, time_elapsed % 60)
        }))

    # eventually creates a folder for this model in the test folder
    test_model_path = test_folder
    if test_folder:
        test_model_path = join(test_folder, model.name)
        makedirs(test_model_path)

    if plot_cmc:
        utils.plot_cmc(y=total_y_closed_set, y_pred_scores=total_y_pred_scores_closed_set,
                       title=model.name, filepath=test_model_path)

    if plot_roc:
        utils.plot_roc_curve(y=total_y_open_set, y_pred_scores=total_y_pred_scores_open_set,
                             title=model.name, filepath=test_model_path)


if __name__ == "__main__":
    parameters_path = join("..", "parameters.json")
    parameters = read_json(filepath=parameters_path)

    # setting the seeds for reproducibility
    seed = parameters["training"]["seed"]
    torch.manual_seed(seed=seed)
    np.random.seed(seed=seed)

    assets_path = join("..", "assets")
    # eventually creates a folder for this test
    test_path = join(assets_path, f"test_{int(time.time())}")
    makedirs(test_path)

    lfw_path, flickr_faces_path = join(assets_path, "lfw"), \
                                  join(assets_path, "flickr_faces")

    models_path = join(assets_path, "models")
    rrdb_pretrained_weights_path = join(models_path, "RRDB_PSNR_x4.pth")

    lfw_dataset_train, lfw_dataset_test = load_lfw_dataset(filepath=assets_path,
                                                           min_faces_per_person=parameters["data"][
                                                               "min_faces_per_person"])
    flickr_dataset = load_flickr_faces_dataset(filepath=flickr_faces_path)
    if len(flickr_dataset) > len(lfw_dataset_test):
        flickr_dataset = flickr_dataset[:len(lfw_dataset_test)]

    flickr_dataloader = DataLoader(dataset=flickr_dataset, shuffle=False,
                                   batch_size=parameters["test"]["batch_size"])
    lfw_dataloader_test = DataLoader(dataset=lfw_dataset_test, shuffle=False,
                                     batch_size=parameters["test"]["batch_size"])
    mixed_dataloader = DataLoader(dataset=lfw_dataset_test + flickr_dataset, shuffle=True,
                                  batch_size=parameters["test"]["batch_size"])
    y = [label.item() for _, label in mixed_dataloader.dataset]

    # plots distribution of labels
    utils.plot_labels_distribution(y=y)

    models = [
        FaceRecognitionModel(name="plain",
                             num_classes=len(set(y) - {-1}),
                             add_noise=False,
                             do_denoising=False,
                             do_super_resolution=False,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        FaceRecognitionModel(name="n",
                             num_classes=len(set(y) - {-1}),
                             add_noise=True,
                             do_denoising=False,
                             do_super_resolution=False,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        FaceRecognitionModel(name="n_dn",
                             num_classes=len(set(y) - {-1}),
                             add_noise=True,
                             do_denoising=True,
                             do_super_resolution=False,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        FaceRecognitionModel(name="SR",
                             num_classes=len(set(y) - {-1}),
                             add_noise=False,
                             do_denoising=False,
                             do_super_resolution=True,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        FaceRecognitionModel(name="n_SR",
                             num_classes=len(set(y) - {-1}),
                             add_noise=True,
                             do_denoising=False,
                             do_super_resolution=True,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        FaceRecognitionModel(name="n_dn_SR",
                             num_classes=len(set(y) - {-1}),
                             add_noise=True,
                             do_denoising=True, denoise_before_sr=True,
                             do_super_resolution=True,
                             noise_prob=parameters["training"]["noise_prob"],
                             resnet_pretrained=True,
                             rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
        # FaceRecognitionModel(name="n_SR_dn",
        #                      num_classes=len(set(y) - {-1}),
        #                      add_noise=True,
        #                      do_denoising=True, denoise_before_sr=False,
        #                      do_super_resolution=True,
        #                      noise_prob=parameters["training"]["noise_prob"],
        #                      resnet_pretrained=True,
        #                      rrdb_pretrained_weights_path=rrdb_pretrained_weights_path),
    ]

    for model in models:
        try:
            model.load_state_dict(torch.load(join(models_path, f"frm_{model.name}.pth")))
        except:
            print(f"No weigths named frm_{model.name}.pth found")
            continue

        test(model, data=mixed_dataloader, resize=True,
             plot_roc=True, plot_cmc=True, test_folder=test_path)
