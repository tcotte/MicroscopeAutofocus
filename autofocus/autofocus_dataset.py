import json
import os
import platform
from typing import Union, List

import numpy as np
import supervisely as sly
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class AutofocusDataset(Dataset):
    def __init__(self, project_dir: str, dataset: str, z_range: Union[List, None] = None, normalize_output=False,
                 transform=None):
        if z_range is None:
            z_range = [-np.inf, np.inf]

        self.normalize_output = normalize_output

        self.transform = transform

        self.project = sly.Project(project_dir, sly.OpenMode.READ)
        self.meta = self.project.meta
        self.dataset = self.project.datasets.get(dataset)

        self.z_range = z_range
        self.images_paths = self.filter_dataset()

        self.img_dir = self.dataset.img_dir
        self.label_dir = self.dataset.ann_dir

    def __len__(self):
        return len(self.images_paths)

    def filter_dataset(self):
        filtered_images = []
        for item in self.dataset.items():
            item_name, picture_path, json_path = item
            annotation = sly.Annotation.load_json_file(json_path, self.meta)
            z_value = annotation.img_tags.get('focus_difference').value
            if self.z_range[0] <= z_value <= self.z_range[1]:
                filtered_images.append(picture_path)

        return filtered_images

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]

        head, tail = os.path.split(img_path)
        annotation = sly.Annotation.load_json_file(os.path.join(self.label_dir, tail + ".json"), self.meta)
        z_value = annotation.img_tags.get('focus_difference').value

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

    path_dataset = r"C:\Users\tristan_cotte\PycharmProjects\prior_controller\output_picture\acquisition_first_slide"
    imgs = list(list_images(path_dataset))
    labels = [get_labelfile_from_imgfile(img) for img in imgs]

    train_dataset = AutofocusDatasetFromList(images_list=imgs, ann_list=labels)

    print(train_dataset[8])
    print(len(train_dataset))
