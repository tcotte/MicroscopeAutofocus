import json
import os
import platform
import time
from timeit import timeit
from typing import Union, List

import cv2
import exif
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


# class AutofocusDataset(Dataset):
#     def __init__(self, project_dir: str, dataset: str, z_range: Union[List, None] = None, normalize_output=False,
#                  transform=None):
#         if z_range is None:
#             z_range = [-np.inf, np.inf]
#
#         self.normalize_output = normalize_output
#
#         self.transform = transform
#
#         self.project = sly.Project(project_dir, sly.OpenMode.READ)
#         self.meta = self.project.meta
#         self.dataset = self.project.datasets.get(dataset)
#
#         self.z_range = z_range
#         self.images_paths = self.filter_dataset()
#
#         self.img_dir = self.dataset.img_dir
#         self.label_dir = self.dataset.ann_dir
#
#     def __len__(self):
#         return len(self.images_paths)
#
#     def filter_dataset(self):
#         filtered_images = []
#         for item in self.dataset.items():
#             item_name, picture_path, json_path = item
#             annotation = sly.Annotation.load_json_file(json_path, self.meta)
#             z_value = annotation.img_tags.get('focus_difference').value
#             if self.z_range[0] <= z_value <= self.z_range[1]:
#                 filtered_images.append(picture_path)
#
#         return filtered_images
#
#     def __getitem__(self, idx):
#         img_path = self.images_paths[idx]
#
#         head, tail = os.path.split(img_path)
#         annotation = sly.Annotation.load_json_file(os.path.join(self.label_dir, tail + ".json"), self.meta)
#         z_value = annotation.img_tags.get('focus_difference').value
#
#         if self.normalize_output:
#             z_value = z_value / self.z_range[1]
#
#         pillow_image = Image.open(img_path)
#
#         if self.transform is None:
#             transform = transforms.ToTensor()
#
#             # Convert the image to PyTorch tensor
#             tensor_image = transform(pillow_image)
#
#         else:
#             transformed = self.transform(image=np.array(pillow_image))
#             tensor_image = transformed["image"]
#
#         return {"X": tensor_image, "y": z_value}

class DifferenceAFDataset(Dataset):
    def __init__(self, excel_filepath: str, image_folder: str, kernel_size: int= 3, transform=None,
                 one_channel_image: bool = False):
        self._excel_filepath = excel_filepath

        self._image_folder = image_folder

        self._df = pd.read_excel(excel_filepath)

        self._kernel_size = kernel_size

        self._transform = transform

        self._one_channel_image = one_channel_image

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx):
        item = self._df.iloc[idx]
        z1_image_name = item['z1_image']
        z2_image_name = item['z2_image']
        xy_position = item['xy_position']

        image_z1_path = os.path.join(self._image_folder, xy_position, z1_image_name)
        image_z2_path = os.path.join(self._image_folder, xy_position, z2_image_name)

        image_z1 = cv2.imread(image_z1_path)
        image_z2 = cv2.imread(image_z2_path)

        blurred_image_z1 = cv2.medianBlur(image_z1, self._kernel_size)
        blurred_image_z2 = cv2.medianBlur(image_z2, self._kernel_size)

        # difference_image = cv2.subtract(blurred_image_z2, blurred_image_z1)
        difference_image = blurred_image_z2 - blurred_image_z1

        blurred_difference_image = np.float32(cv2.medianBlur(difference_image, self._kernel_size))

        y = float(item['z2_diff_focus'])

        norm_image = self.normalize_standard_channelwise(img_np=blurred_difference_image)

        if self._one_channel_image:
            norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)

        if self._transform is None:
            transform = transforms.ToTensor()

            # Convert the image to PyTorch tensor
            tensor_image = transform(norm_image)

        else:
            transformed = self._transform(image=norm_image)
            tensor_image = transformed["image"]

        return {'X': tensor_image, 'y': y, 'std': torch.tensor(self.luminance_weighted_std(img_np=image_z1))}

    @staticmethod
    def luminance_weighted_std(img_np: np.ndarray) -> float:
        std_image = 0
        # [0.114, 0.587, 0.299] -> BGR luminance weights
        for c, coeff in zip(range(img_np.shape[2]), [0.114, 0.587, 0.299]):
            channel = img_np[..., c]
            std = channel.std()
            std_image += std * coeff

        return std_image

    @staticmethod
    def transform_to_grayscale(img_np: np.ndarray) -> np.ndarray:

        grayscale_image = np.zeros(img_np.shape[:2])
        luminance_weights = [0.114, 0.587, 0.299]

        for index, weight in enumerate(luminance_weights):
            grayscale_image += weight * img_np[..., index]

        return grayscale_image

    @staticmethod
    def normalize_standard_channelwise(img_np):
        norm = np.zeros_like(img_np)
        for c in range(img_np.shape[2]):
            channel = img_np[..., c]
            mean, std = channel.mean(), channel.std()
            norm[..., c] = (channel - mean) / std
        return norm


class AutofocusDatasetFromMetadata(Dataset):
    def __init__(self, images_list: List[str], z_range: Union[List, None] = None, normalize_output=False,
                 transform=None):
        if z_range is None:
            z_range = [-np.inf, np.inf]

        self.images_list = images_list
        self.normalize_output = normalize_output

        self.transform = transform

        self.z_range = z_range

    def __len__(self):
        return len(self.images_list)

    @staticmethod
    def get_focus_diff_from_exif_metadata(img_path: str) -> float:
        return float(exif.Image(img_path).make)

    def __getitem__(self, idx):
        img_path = self.images_list[idx]
        z_value = self.get_focus_diff_from_exif_metadata(img_path=img_path)

        if self.normalize_output:
            z_value = z_value / self.z_range[1]

        pillow_image = Image.open(img_path)

        if self.transform is None:
            transform = transforms.ToTensor()

            # Convert the image to PyTorch tensor
            tensor_image = transform(pillow_image)

        else:
            transformed = self.transform(image=np.array(pillow_image))
            tensor_image = transformed["image"]

        return {"X": tensor_image, "y": z_value}


class AutofocusDatasetFromList(Dataset):
    def __init__(self, images_list: List[str], ann_list: List[str], z_range: Union[List, None] = None,
                 normalize_output=False,
                 transform=None):
        if z_range is None:
            z_range = [-np.inf, np.inf]

        self.images_list = images_list
        self.ann_list = ann_list

        self.normalize_output = normalize_output

        self.transform = transform

        self.z_range = z_range

    def __len__(self):
        return len(self.images_list)

    @staticmethod
    def get_focus_diff_from_json(annotation_file):
        with open(annotation_file) as json_file:
            data = json.load(json_file)

        for i in data["tags"]:
            if i.get("name") == "focus_difference":
                return i.get("value")

        raise "Focus difference tags was not found"

    def __getitem__(self, idx):
        img_path = self.images_list[idx]
        z_value = self.get_focus_diff_from_json(annotation_file=self.ann_list[idx])

        if self.normalize_output:
            z_value = z_value / self.z_range[1]

        pillow_image = Image.open(img_path)

        if self.transform is None:
            transform = transforms.ToTensor()

            # Convert the image to PyTorch tensor
            tensor_image = transform(pillow_image)

        else:
            transformed = self.transform(image=np.array(pillow_image))
            tensor_image = transformed["image"]

        return {"X": tensor_image, "y": z_value}


def get_labelfile_from_imgfile(img_path):
    path = os.path.normpath(img_path)
    splitted_path = path.split(os.sep)
    if get_os() == "Windows":
        label_path = os.path.join("C:\\", *splitted_path[1:-2], "ann", splitted_path[-1] + ".json")
        return label_path
    else:
        label_path = os.path.join(*splitted_path[2:-2], "ann", splitted_path[-1] + ".json")
        return os.path.abspath(label_path)


def get_os() -> str:
    return platform.system()


if __name__ == "__main__":
    from imutils.paths import list_images

    # path_dataset = r"C:\Users\tristan_cotte\PycharmProjects\microscope_autofocus\autofocus\data\dataset_v1\X\slide_3"
    # imgs = list(list_images(path_dataset))
    # labels = [get_labelfile_from_imgfile(img) for img in imgs]
    #
    # train_dataset = AutofocusDatasetFromList(images_list=imgs, ann_list=labels)
    #
    # print(train_dataset[8])
    # print(len(train_dataset))
    import matplotlib.pyplot as plt

    ds = DifferenceAFDataset(excel_filepath=r'data/diff_train.xlsx',
                             image_folder=r'data\dataset_09_25_2025\X\train')
    image, y = ds[0]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image, cmap='gray', vmin=-1, vmax=1)
    plt.show()
