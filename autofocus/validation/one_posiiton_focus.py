import os.path

import albumentations as A
import numpy as np
import torch
import torchvision
from imutils.paths import list_images
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from autofocus.autofocus_dataset import get_labelfile_from_imgfile, AutofocusDatasetFromMetadata
from autofocus.utils import get_device

BATCH_SIZE = 64
# Augmentations
from torch.utils.data import DataLoader

class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def compute_rmse(y_hat, y_ground_truth):
    try:
        mse = np.square(np.subtract(np.array(y_ground_truth), np.array(y_hat))).mean()
        rmse = np.sqrt(mse)

    except Exception as e:
        print(str(e))
    return rmse

# train_transform = A.Compose([
#     A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
#     A.pytorch.transforms.ToTensorV2(),
# ])

test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
    A.pytorch.transforms.ToTensorV2()
])

# Pytorch datasets
# test_dataset = AutofocusDataset(
#     project_dir=r"C:\Users\tristan_cotte\PycharmProjects\prior_controller\output_picture\dataset_Z\slide5",
#     dataset="68610x_26972y", transform=test_transform)
# test_dataset = AutofocusDataset(
#     project_dir=r"C:\Users\tristan_cotte\PycharmProjects\prior_controller\autofocus\sly_project",
#     dataset="ds1", transform=test_transform)
# path_dataset = os.path.join(r"D:\03 - IDEA\Micronoyaux\Autofocus\datasets_v2 - Copy\X\test\67471x_16896y")
# imgs = list(list_images(path_dataset))
# labels = [get_labelfile_from_imgfile(img) for img in imgs]
# test_dataset = AutofocusDatasetFromMetadata(images_list=imgs, transform=test_transform)
#
# # Dataloaders
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
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

    model_checkpoint = torch.load(r'C:\Users\tristan_cotte\Downloads\100th_epoch_chkpt.pt')
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    path_test_dataset = r'D:\03 - IDEA\Micronoyaux\Autofocus\dataset_09_25_2025\X\test'

    unnormalizer = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    for position in os.listdir(path_test_dataset):
        path_dataset = os.path.join(path_test_dataset, position)
        imgs = list(list_images(path_dataset))
        labels = [get_labelfile_from_imgfile(img) for img in imgs]
        test_dataset = AutofocusDatasetFromMetadata(images_list=imgs, transform=test_transform)

        # Dataloaders
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


        y = []
        y_hat = []

        with torch.no_grad():
            for idx in tqdm(range(len(test_dataset))):
                # data = torch.unsqueeze(test_dataset[idx]["X"].float(), dim=0)
                data = torch.unsqueeze(test_dataset[idx]["X"], dim=0)
                data_visible = test_dataset[idx]["X"].permute(1, 2, 0)
                # plt.imshow(data_visible)
                # plt.imshow(cv2.cvtColor(data_visible, cv2.COLOR_BGR2RGB))
                # plt.show()
                data = data.to(device)
                y_hat.append(model(data).cpu().item())
                y.append(test_dataset[idx]["y"])

                if test_dataset[idx]["y"] == 0:
                    focus_image = unnormalizer(data)

        try:
            rmse = compute_rmse(y_hat=y_hat, y_ground_truth=y)
        except:
            print("Can not compute rmse")

        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        plt.title(position)
        ax0.set_title(f"RMSE {rmse:.2f}")
        ax0.plot(y, y)
        ax0.scatter(y, y_hat, c='r')
        ax0.set_xlabel('Z distance from focus (µm)')
        ax0.set_ylabel('Predicted Z distance from focus (µm)')
        ax1.imshow(torch.squeeze(focus_image).permute(1, 2, 0).cpu().numpy())
        plt.show()
        # plt.savefig(os.path.join(r'C:\Users\tristan_cotte\PycharmProjects\microscope_autofocus\output_validation\test',
        #                          f'{position}.png'))
