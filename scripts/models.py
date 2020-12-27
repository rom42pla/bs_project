from typing import Union

import torch
from PIL.Image import NEAREST
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms


# base class for each custom module
class CustomModule(nn.Module):
    def __init__(self, device: str = "auto"):
        super(CustomModule, self).__init__()
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} \
            else "cuda" if torch.cuda.is_available() else "cpu"


class SaltAndPepperNoise(CustomModule):
    def __init__(self, noise_prob_per_pixel: float = 0.005,
                 device: str = "auto"):
        super(SaltAndPepperNoise, self).__init__(device=device)
        assert isinstance(noise_prob_per_pixel, float) and 0 <= noise_prob_per_pixel <= 1
        self.prob = noise_prob_per_pixel

        self.to(self.device)

    def forward(self, X):
        mask = torch.rand(size=X.shape).to(X.device) >= self.prob
        return X * mask


class Scaler(CustomModule):
    def __init__(self, size: Union[int, tuple]):
        super(Scaler, self).__init__()

        self.size = size

    def forward(self, X: torch.Tensor):
        out = transforms.Resize(self.size, interpolation=NEAREST)(X)
        return out


class Classifier(CustomModule):
    def __init__(self, num_classes: int, pretrained: bool = True,
                 normalize_input: bool = False,
                 device: str = "auto"):
        super(Classifier, self).__init__(device=device)

        assert isinstance(num_classes, int) and num_classes >= 2
        assert isinstance(pretrained, bool)
        assert isinstance(normalize_input, bool)

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
        # X = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])(X)
        out = self.layers(X)

        if not self.training:
            out = F.log_softmax(out, dim=-1)
        return out


class RRDB(CustomModule):
    def __init__(self, pretrained_weights_path: str, trainable: bool = False,
                 device: str = "auto"):
        super(RRDB, self).__init__(device=device)

        # checks that the weights are correctly given
        assert isinstance(pretrained_weights_path, str)

        from assets.models.RRDBNet_arch import RRDBNet
        rrdb = RRDBNet(3, 3, 64, 23, gc=32)
        if pretrained_weights_path:
            rrdb.load_state_dict(torch.load(pretrained_weights_path), strict=True)
        if trainable:
            for parameter in rrdb.parameters():
                parameter.requires_grad = False
            for parameter in rrdb.trunk_conv.parameters():
                parameter.requires_grad = True
            rrdb.train()
        else:
            for parameter in rrdb.parameters():
                parameter.requires_grad = False
            rrdb.eval()

        self.layers = nn.Sequential(
            rrdb
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        out = self.layers(X)
        return out

class Denoiser(CustomModule):
    def __init__(self, trainable: bool = False,
                 device: str = "auto"):
        super(Denoiser, self).__init__(device=device)

        from assets.models.dncnn import DnCNN
        denoiser = DnCNN()
        for parameter in denoiser.parameters():
            parameter.requires_grad = trainable


        self.layers = nn.Sequential(
            denoiser
        )

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        out = self.layers(X)
        return out

class FaceRecognitionModel(CustomModule):
    def __init__(self, num_classes: int,
                 add_noise: bool = False, noise_prob: float = 0.005,
                 do_denoising: bool = True,
                 do_super_resolution: bool = False,
                 rrdb_pretrained_weights_path: str = None, resnet_pretrained: bool = True,
                 device: str = "auto"):
        super(FaceRecognitionModel, self).__init__(device=device)

        assert isinstance(num_classes, int) and num_classes >= 2
        self.num_classes = num_classes

        # checks that the weights are correctly given
        assert isinstance(resnet_pretrained, bool)

        # noise parameters
        assert isinstance(add_noise, bool)
        self.add_noise = add_noise
        if self.add_noise:
            self.noise_adding = SaltAndPepperNoise(noise_prob_per_pixel=noise_prob)
        assert isinstance(do_denoising, bool)
        self.do_denoising = do_denoising
        if self.do_denoising:
            self.denoiser = Denoiser()

        # super resolution parameters
        assert isinstance(do_super_resolution, bool)
        self.do_super_resolution = do_super_resolution
        if self.do_super_resolution:
            assert isinstance(rrdb_pretrained_weights_path, str)
            self.super_resolution_model = RRDB(pretrained_weights_path=rrdb_pretrained_weights_path)

        self.classifier = Classifier(pretrained=resnet_pretrained, num_classes=num_classes)

        # moves the entire model to the chosen device
        self.to(self.device)

    def forward(self, X: torch.Tensor):
        if self.add_noise:
            X = self.noise_adding(X)
        if self.do_denoising:
            X = self.denoiser(X)
        if self.do_super_resolution:
            X = self.super_resolution_model(X)
        scores = self.classifier(X)
        return scores
