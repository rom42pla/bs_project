from os.path import join

from utils import load_lfw_dataset, split_dataset
from torchvision import transforms

assets_path = join("..", "assets")
lfw_path = join(assets_path, "lfw")

transform = transforms.Compose([
    transforms.Resize(250),
    transforms.CenterCrop(250),
    transforms.ToTensor()
])
lfw_dataset = load_lfw_dataset(filepath=assets_path, transform=transform)
lfw_subset_train, lfw_subset_val, lfw_subset_test = split_dataset(lfw_dataset, [0.8, 0.1, 0.1])








