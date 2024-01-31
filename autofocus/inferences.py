import os.path

import albumentations as A
import albumentations.pytorch
import numpy as np
import torch
import torchvision
from imutils.paths import list_images
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from utils import get_device

BATCH_SIZE = 64
# Augmentations
from torch.utils.data import DataLoader

from autofocus_dataset import get_labelfile_from_imgfile, AutofocusDatasetFromList


def rmse(y_hat, y_ground_truth):
    mse = np.square(np.subtract(np.array(y_ground_truth), np.array(y_hat))).mean()
    return np.sqrt(mse)

# train_transform = A.Compose([
#     A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
#     A.pytorch.transforms.ToTensorV2(),
# ])

test_transform = A.Compose([
    A.Normalize(),
    A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
    A.pytorch.transforms.ToTensorV2()
])

# Pytorch datasets
# train_dataset = AutofocusDataset(
#     project_dir=r"C:\Users\tristan_cotte\PycharmProjects\prior_controller\output_picture\dataset_Z\slide5",
#     dataset="68610x_26972y", transform=test_transform)
# test_dataset = AutofocusDataset(
#     project_dir=r"C:\Users\tristan_cotte\PycharmProjects\prior_controller\autofocus\sly_project",
#     dataset="ds1", transform=test_transform)
path_dataset = os.path.join(r"C:\Users\tristan_cotte\PycharmProjects\prior_controller\output_picture\dataset_Z_sly",
                            "15408x_25564y")
imgs = list(list_images(path_dataset))
labels = [get_labelfile_from_imgfile(img) for img in imgs]
train_dataset = AutofocusDatasetFromList(images_list=imgs, ann_list=labels, transform=test_transform)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

### Model

# CNN regression model
model = torchvision.models.mobilenet_v3_small()

layers = []
layers += [nn.Linear(in_features=576, out_features=1024)]
layers += [nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
layers += [nn.Dropout(p=0.5)]
layers += [nn.Linear(1024, 512, bias=True), nn.Hardswish(inplace=True)]
layers += [nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
layers += [nn.Dropout(p=0.5)]
layers += [nn.Linear(512, 16, bias=True), nn.Hardswish(inplace=True)]
layers += [nn.Linear(16, 1)]
model.classifier = nn.Sequential(*layers)

if __name__ == "__main__":
    device = get_device()

    model = torch.load(r"C:\Users\tristan_cotte\PycharmProjects\microscope_autofocus\autofocus\checkpoint\run_fixed_large_dataset_last.pt")
    model.to(device)
    model.eval()


    y = []
    y_hat = []

    with torch.no_grad():
        for idx in tqdm(range(len(train_dataset))):
            # data = torch.unsqueeze(train_dataset[idx]["X"].float(), dim=0)
            data = torch.unsqueeze(train_dataset[idx]["X"], dim=0)
            data_visible = train_dataset[idx]["X"].permute(1, 2, 0)
            # plt.imshow(data_visible)
            # plt.imshow(cv2.cvtColor(data_visible, cv2.COLOR_BGR2RGB))
            # plt.show()
            data = data.to(device)
            y_hat.append(model(data).cpu().item())
            y.append(train_dataset[idx]["y"])

    rmse = rmse(y_hat=y_hat, y_ground_truth=y)

    plt.title(f"RMSE {rmse:.2f}")
    plt.plot(y, y)
    plt.scatter(y, y_hat, c='r')
    plt.xlabel('Z distance from focus (µm)')
    plt.ylabel('Predicted Z distance from focus (µm)')
    plt.show()


