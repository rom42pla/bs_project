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
lfw_dataloader_train, lfw_dataloader_val, lfw_dataloader_test = split_dataset(lfw_dataset, splits=[0.8, 0.1, 0.1],
                                                                              batch_size=30)

for dataloader in [lfw_dataloader_train, lfw_dataloader_val, lfw_dataloader_test]:
    print(len(dataloader))










