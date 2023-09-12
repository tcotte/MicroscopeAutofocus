import albumentations as A
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import Linear, Hardswish, Dropout

BATCH_SIZE = 64
# Augmentations
from torch.utils.data import DataLoader

from autofocus.autofocus_dataset import AutofocusDataset

train_transform = A.Compose([
    A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.pytorch.transforms.ToTensorV2(),
])

test_transform = A.Compose([
    A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
    A.pytorch.transforms.ToTensorV2()
])

# Pytorch datasets
train_dataset = AutofocusDataset(
    project_dir=r"C:\Users\tristan_cotte\PycharmProjects\prior_controller\autofocus\sly_project",
    dataset="ds0", transform=test_transform)
test_dataset = AutofocusDataset(
    project_dir=r"C:\Users\tristan_cotte\PycharmProjects\prior_controller\autofocus\sly_project",
    dataset="ds1", transform=test_transform)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

### Model

# CNN regression model
model = torchvision.models.mobilenet_v3_small()
model.classifier = nn.Sequential(Linear(in_features=576, out_features=1024),
                                 Hardswish(),
                                 Dropout(p=0.2),
                                 Linear(in_features=1024, out_features=1))

if __name__ == "__main__":
    model.load_state_dict(torch.load("model_autofocus.pt"))
    model.eval()

    y = []
    y_hat = []

    with torch.no_grad():
        for idx in range(200, 300):
            y_hat.append(model(torch.unsqueeze(train_dataset[idx]["X"].float(), dim=0)).item())
            y.append(train_dataset[idx]["y"])

    print(len(y), len(y_hat))
    plt.plot(y, y)
    plt.scatter(y_hat, y, c='r')
    plt.show()
