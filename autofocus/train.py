"""
Regression accuracies: https://machinelearningmastery.com/regression-metrics-for-machine-learning/
Deep learning autofocus: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8803042/#r24
"""
import argparse
import os
from typing import Optional

import albumentations as A
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf

from autofocus.autofocus_dataset import AutofocusDataset
from autofocus.autofocus_model import RegressionMobilenet
from autofocus.logger import WeightandBiaises
from autofocus.utils import get_device, get_os

parser = argparse.ArgumentParser(
    prog='Autofocus on microscope',
    description='This program enables to train a model which is able to know from a picture at which '
                'distance from the focus the microscope is.',
    epilog='--- Tristan COTTE --- SGS France Excellence Opérationnelle ---')
parser.add_argument('-epoch', '--epoch', type=int, default=100, required=False,
                    help='Number of epochs used for train the model')
parser.add_argument('-device', '--device', type=str, default="cuda", required=False,
                    help='Device used to train the model')
parser.add_argument('-src', '--source_project', type=str, required=True,
                    help='Path of Supervisely root project which contains pictures')
parser.add_argument('-trs', '--train_set', type=str, required=True,
                    help='Dataset of train images')
parser.add_argument('-tes', '--test_set', type=str, required=True,
                    help='Dataset of test images')
parser.add_argument('-wd', '--weight_decay', type=float, default=0, required=False,
                    help='Weight decay used to regularized')
parser.add_argument('-bs', '--batch_size', type=int, default=64, required=False,
                    help='Batch size during the training')
parser.add_argument('-sz', '--img_size', type=int, default=512, required=False,
                    help='Training img size')
parser.add_argument('-do', '--dropout', type=float, default=0.2, required=False,
                    help='Dropout used for the training')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, required=False,
                    help='Learning rate used for training')
parser.add_argument('-project', '--project_name', type=str, default="unet_project", required=False,
                    help='Name of the project in W&B')
parser.add_argument('-name', '--run_name', type=Optional[str], default=None, required=False,
                    help='Name of the run in W&B')

parser.add_argument('-display', '--interval_display', type=int, default=10, required=False,
                    help='Interval of display mask in W&B')

# pretrain model
parser.add_argument("-weights", "--pretrained_weights", default=False, action="store_true", required=False,
                    help="Use pretrained weights")

args = parser.parse_args()

### Datasets

# Augmentations
train_transform = A.Compose([
    A.Normalize(),
    A.augmentations.geometric.resize.LongestMaxSize(max_size=args.img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.augmentations.transforms.PixelDropout (dropout_prob=0.01),
    A.RandomBrightnessContrast(p=0.2),
    A.pytorch.transforms.ToTensorV2(),
])

test_transform = A.Compose([
    A.augmentations.geometric.resize.LongestMaxSize(max_size=args.img_size),
    A.pytorch.transforms.ToTensorV2()
])

# Pytorch datasets
train_dataset = AutofocusDataset(
    project_dir=args.source_project,
    dataset=args.train_set, transform=train_transform, z_range=[-145, 150])
test_dataset = AutofocusDataset(
    project_dir=args.source_project,
    dataset=args.test_set, transform=test_transform, z_range=[-145, 150])

# Dataloaders
if get_os().lower() == "windows":
    num_workers = 0
else:
    num_workers = os.cpu_count()

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=num_workers)

### Model

pretrained_weights = ("pretrained", MobileNet_V3_Small_Weights.IMAGENET1K_V1)

if args.pretrained_weights:
    pretrained_weights = ("pretrained", MobileNet_V3_Small_Weights.IMAGENET1K_V1)
else:
    pretrained_weights = None

model = RegressionMobilenet(*_mobilenet_v3_conf("mobilenet_v3_small"), dropout=args.dropout, weights=pretrained_weights)

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

### Training

if __name__ == "__main__":

    device = get_device()
    conf= {"device": device, "loss": str(criterion), "optimizer": str(optimizer), "lr": args.learning_rate,
           "weight_decay": args.weight_decay, "pretrained_model": args.pretrained_weights,
           "batch_size": args.batch_size, "nb_epoch": args.epoch, "dropout": args.dropout, "img_size": args.img_size}

    w_b = WeightandBiaises(project_name=args.project_name, run_id=args.run_name, config=conf)

    nb_train_batch = np.ceil(len(train_dataset) / args.batch_size)
    nb_test_batch = np.ceil(len(test_dataset) / args.batch_size)

    mse_func = nn.MSELoss(reduction="sum")


    model.to(device)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(args.epoch):  # loop over the dataset multiple times
        train_running_loss = 0.0
        test_running_loss = 0.0
        train_mse = 0.0
        test_mse = 0.0

        model.train()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data["X"].float(), data["y"]

            images = images.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            train_loss = criterion(outputs.squeeze(), labels)
            train_loss.backward()
            optimizer.step()

            train_mse += mse_func(outputs.squeeze(), labels)

            # print statistics
            train_running_loss += train_loss.item()

        model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data["X"].float(), data["y"]

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                test_mse = mse_func(outputs.squeeze(), labels)
                test_loss = criterion(outputs.squeeze(), labels)

                test_running_loss += test_loss.item()

                w_b.log_table(outputs.squeeze(), images, labels, epoch)

        w_b.log_mse(train_mse=train_mse/len(train_dataset), test_mse=test_mse/len(test_dataset), epoch=epoch)
        w_b.log_losses(train_loss=train_running_loss, test_loss=test_running_loss, epoch=epoch)

        print(f"Epoch {str(epoch + 1)}: train_loss {train_running_loss} -- test_loss {test_running_loss} -- "
              f"train_accuracy {train_mse.item()/len(train_dataset)} -- "
              f"test_accuracy {test_mse.item()/len(test_dataset)}")

        train_losses.append(train_running_loss)
        test_losses.append(test_running_loss)
        train_accuracies.append(train_mse.item()/len(train_dataset))
        test_accuracies.append(test_mse.item()/len(test_dataset))

    torch.save(model.state_dict(), args.run_name + ".pt")
    w_b.save_model(model_name="last.pt", model=model)

    # Plot training data
    t = list(range(args.epoch + 1))
    _, axs = plt.subplots(1, 2, layout='constrained')
    axs[0].plot(t, train_accuracies, 'b', label="train_accuracies")
    axs[0].plot(t, test_accuracies, 'r', label="test_accuracies")
    axs[0].set_title("MSE during training")
    axs[0].set_ylabel("MSE")
    axs[0].set_xlabel("Epochs")
    axs[0].legend()

    axs[1].plot(t, train_losses, 'b', label="train_losses")
    axs[1].plot(t, test_losses, 'r', label="test_losses")
    axs[1].set_title("Losses during training")
    axs[1].set_ylabel("L1 smooth loss")
    axs[1].set_xlabel("Epochs")
    axs[1].legend()

    plt.show()
