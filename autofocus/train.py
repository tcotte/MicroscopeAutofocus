"""
Regression accuracies: https://machinelearningmastery.com/regression-metrics-for-machine-learning/
Deep learning autofocus: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8803042/#r24
"""
import argparse
from typing import Optional

import albumentations as A
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import Linear, Hardswish, Dropout
from torch.utils.data import DataLoader
from autofocus.autofocus_dataset import AutofocusDataset

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

parser.add_argument('-bs', '--batch_size', type=int, default=64, required=False,
                    help='Batch size during the training')
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
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
NB_EPOCHS = args.epoch

### Datasets

# Augmentations
train_transform = A.Compose([
    A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.augmentations.transforms.PixelDropout (dropout_prob=0.01),
    A.RandomBrightnessContrast(p=0.2),
    A.pytorch.transforms.ToTensorV2(),
])

test_transform = A.Compose([
    A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
    A.pytorch.transforms.ToTensorV2()
])

# Pytorch datasets
train_dataset = AutofocusDataset(
    project_dir=args.source_project,
    dataset=args.train_set, transform=train_transform)
test_dataset = AutofocusDataset(
    project_dir=args.source_project,
    dataset=args.test_set, transform=test_transform)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

### Model

# CNN regression model
if args.pretrained_weights:
    model = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
else:
    model = torchvision.models.mobilenet_v3_small()
model.classifier = nn.Sequential(Linear(in_features=576, out_features=1024),
                                 Hardswish(),
                                 Dropout(p=0.2),
                                 Linear(in_features=1024, out_features=1))

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

### Training

if __name__ == "__main__":
    nb_train_batch = np.ceil(len(train_dataset) / BATCH_SIZE)
    nb_test_batch = np.ceil(len(test_dataset) / BATCH_SIZE)

    mse_func = nn.MSELoss(reduction="mean")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(NB_EPOCHS):  # loop over the dataset multiple times
        train_running_loss = 0.0
        test_running_loss = 0.0
        train_mse = 0.0
        test_mse = 0.0

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
            if i % 2 == 0:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss.item():.3f}  -- MSE: {train_mse}')

        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data["X"].float(), data["y"]

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                test_mse = mse_func(outputs.squeeze(), labels)
                test_loss = criterion(outputs.squeeze(), labels)

                test_running_loss += test_loss.item()

        print(f"Epoch {str(epoch + 1)}: train_loss {train_running_loss} -- test_loss {test_running_loss} -- "
              f"train_accuracy {train_mse.item() / nb_train_batch} -- test_accuracy {test_mse.item() / nb_test_batch}")

        train_losses.append(train_running_loss)
        test_losses.append(test_running_loss)
        train_accuracies.append(train_mse.item() / nb_train_batch)
        test_accuracies.append(test_mse.item() / nb_test_batch)

    torch.save(model.state_dict(), "model_autofocus_1.pt")

    t = list(range(NB_EPOCHS))
    _, axs = plt.subplots(1, 2, layout='constrained')
    axs[0].plot(t, train_accuracies, 'b', label="train_accuracies")  # plotting t, a separately
    axs[0].plot(t, test_accuracies, 'r', label="test_accuracies")
    axs[0].set_title("MSE during training")
    axs[0].set_ylabel("MSE")
    axs[0].set_xlabel("Epochs")
    axs[0].legend()

    axs[1].plot(t, train_losses, 'b', label="train_losses")  # plotting t, a separately
    axs[1].plot(t, test_losses, 'r', label="test_losses")
    axs[1].set_title("Losses during training")
    axs[1].set_ylabel("L1 smooth loss")
    axs[1].set_xlabel("Epochs")
    axs[1].legend()

    plt.show()
