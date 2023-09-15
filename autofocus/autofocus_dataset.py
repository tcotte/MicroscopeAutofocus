import os
from typing import Union, List

import numpy as np
import supervisely as sly
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class AutofocusDataset(Dataset):
    def __init__(self, project_dir: str, dataset: str, z_range: Union[List, None] = None, transform=None):
        if z_range is None:
            z_range = [-np.inf, np.inf]

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

        pillow_image = Image.open(img_path)

        if self.transform is None:
            transform = transforms.ToTensor()

            # Convert the image to PyTorch tensor
            tensor_image = transform(pillow_image)

        else:
            transformed = self.transform(image=np.array(pillow_image))
            tensor_image = transformed["image"]

        return {"X": tensor_image, "y": z_value}
