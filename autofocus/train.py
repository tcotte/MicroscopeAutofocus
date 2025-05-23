"""
Regression accuracies: https://machinelearningmastery.com/regression-metrics-for-machine-learning/
Deep learning autofocus: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8803042/#r24
"""
import argparse
import logging
import os

import albumentations as A
import numpy as np
import torch
import torchvision
from imutils.paths import list_images
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights
from tqdm import tqdm
from autofocus_dataset import AutofocusDatasetFromList, AutofocusDatasetFromMetadata
from logger import WeightandBiaises
from utils import get_device, get_os

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
parser.add_argument('-trs', '--train_set', type=str, required=False,
                    help='Dataset of train images')
parser.add_argument('-tes', '--test_set', type=str, required=False,
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
parser.add_argument('-project', '--project_name', type=str, default="Microscope_autofocus", required=False,
                    help='Name of the project in W&B')
parser.add_argument('-name', '--run_name', type=str, default=None, required=False,
                    help='Name of the run in W&B')
parser.add_argument('-display', '--interval_display', type=int, default=10, required=False,
                    help='Interval of display mask in W&B')
parser.add_argument('-z', '--z_range', nargs='+', help='Picture selection filtered in Z range', required=False)
parser.add_argument("-weights", "--pretrained_weights", default=False, action="store_true", required=False,
                    help="Use pretrained weights")
parser.add_argument("--split_by_xy_positions", default=False, action="store_true", required=False,
                    help="Instead of split dataset picture by picture, split the dataset depending on XY position")
parser.add_argument("-norm", "--normalize_output", default=False, action="store_true", required=False,
                    help="Normalize output in range [-1;1]")

args = parser.parse_args()

### Fix seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

### Datasets
# Augmentations
train_transform = A.Compose([
    A.Normalize(),
    A.augmentations.geometric.resize.LongestMaxSize(max_size=args.img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1)]),
    A.PixelDropout(dropout_prob=0.01),
    A.RandomBrightnessContrast(p=0.2),
    A.pytorch.transforms.ToTensorV2(),
])

test_transform = A.Compose([
    A.Normalize(),
    A.augmentations.geometric.resize.LongestMaxSize(max_size=args.img_size),
    A.pytorch.transforms.ToTensorV2()
])

# Pytorch datasets
if args.train_set is not None:
    if args.z_range is not None:
        args.z_range = [int(i) for i in args.z_range]
    train_dataset = AutofocusDataset(
        project_dir=args.source_project,
        dataset=args.train_set, transform=train_transform, z_range=args.z_range,
        normalize_output=args.normalize_output)
    test_dataset = AutofocusDataset(
        project_dir=args.source_project,
        dataset=args.test_set, transform=test_transform, z_range=args.z_range,
        normalize_output=args.normalize_output)

else:
    logging.info('Split training and test sets...')
    X = list(list_images(args.source_project))
    if not args.split_by_xy_positions:
        y = len(list(list_images(args.source_project)))*[0] # fake y to compute train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    else:
        xy_positions = list(set([os.path.basename(os.path.dirname(i)) for i in X]))
        train_positions, test_positions, _, _ = train_test_split(xy_positions, [0] * len(xy_positions), test_size=0.2,
                                                                 random_state=42)
        X_train = []
        X_test = []
        for element in X:
            if os.path.basename(os.path.dirname(element)) in train_positions:
                X_train.append(element)
            else:
                X_test.append(element)

        y_train = [0]*len(X_train)
        y_test = [0]*len(X_test)

    logging.info('Datasets creation...')
    train_dataset = AutofocusDatasetFromMetadata(images_list=X_train, transform=train_transform)
    test_dataset = AutofocusDatasetFromMetadata(images_list=X_test, transform=test_transform)


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

if pretrained_weights:
    model = torchvision.models.mobilenet_v3_small(weights='DEFAULT', dropout=args.dropout)
else:
    model = torchvision.models.mobilenet_v3_small(dropout=args.dropout)

# https://medium.com/analytics-vidhya/fastai-image-regression-age-prediction-based-on-image-68294d34f2ed

layers = []
layers += [nn.Linear(in_features=576, out_features=1024)]
layers += [nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
layers += [nn.Dropout(p=args.dropout)]
layers += [nn.Linear(1024, 512, bias=True), nn.Hardswish(inplace=True)]
layers += [nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
layers += [nn.Dropout(p=args.dropout)]
layers += [nn.Linear(512, 16, bias=True), nn.Hardswish(inplace=True)]
layers += [nn.Linear(16, 1)]
model.classifier = nn.Sequential(*layers)

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

### Training

if __name__ == "__main__":

    device = get_device()
    conf = {"device": device, "loss": str(criterion), "optimizer": str(optimizer), "lr": args.learning_rate,
            "weight_decay": args.weight_decay, "pretrained_model": args.pretrained_weights,
            "batch_size": args.batch_size, "nb_epoch": args.epoch, "dropout": args.dropout, "img_size": args.img_size}

    w_b = WeightandBiaises(project_name=args.project_name, run_id=args.run_name, config=conf)

    nb_train_batch = int(np.ceil(len(train_dataset) / args.batch_size))
    nb_test_batch = int(np.ceil(len(test_dataset) / args.batch_size))

    mse_func = nn.L1Loss(reduction="sum")

    model.to(device)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(args.epoch):  # loop over the dataset multiple times
        train_running_loss = 0.0
        test_running_loss = 0.0
        train_mae = 0.0
        test_mae = 0.0

        model.train()

        with tqdm(train_dataloader, unit="batch") as t_epoch:
            for batch in t_epoch:
                t_epoch.set_description(f"Epoch {epoch}")

                # get the inputs; data is a list of [inputs, labels]
                images, labels = batch["X"].float(), batch["y"]

                images = images.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(images)
                train_loss = criterion(outputs.squeeze(), labels)
                train_loss.backward()
                optimizer.step()

                train_mae += mse_func(outputs.squeeze(), labels)

                # print statistics
                train_running_loss += train_loss.item()

        model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data["X"].float(), data["y"]

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                test_mae += mse_func(outputs.squeeze(), labels)
                test_loss = criterion(outputs.squeeze(), labels)

                test_running_loss += test_loss.item()

        if not args.normalize_output:
            w_b.log_table(outputs.squeeze(), images, labels, epoch + 1)
        else:
            w_b.log_table(outputs.squeeze() * int(args.z_range[1]), images, labels * int(args.z_range[1]), epoch + 1)
            train_mae = train_mae.item() * int(args.z_range[1])
            test_mae = test_mae.item() * int(args.z_range[1])

        train_running_loss = train_running_loss / nb_train_batch
        test_running_loss = test_running_loss / nb_test_batch

        w_b.log_mae(train_mse=train_mae / len(train_dataset), test_mse=test_mae / len(test_dataset), epoch=epoch + 1)
        w_b.log_losses(train_loss=train_running_loss, test_loss=test_running_loss, epoch=epoch + 1)

        print(f"Epoch {str(epoch + 1)}: train_loss {train_running_loss} -- test_loss {test_running_loss} -- "
              f"train_accuracy {train_mae / len(train_dataset)} -- "
              f"test_accuracy {test_mae / len(test_dataset)}")

        train_losses.append(train_running_loss)
        test_losses.append(test_running_loss)
        train_accuracies.append(train_mae / len(train_dataset))
        test_accuracies.append(test_mae / len(test_dataset))

        if epoch % 10 == 0:
            w_b.save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, train_loss=train_running_loss,
                                test_loss=test_running_loss)

    torch.save(model.state_dict(), args.run_name + ".pt")
    w_b.save_model(model_name="last.pt", model=model)

    print("[SUCCESS]: model was trained without disagreement.")