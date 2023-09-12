import os

import albumentations as A
import albumentations.pytorch
import imutils.paths
import numpy as np
import supervisely as sly
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class AutofocusDataset(Dataset):
    def __init__(self, project_dir: str, dataset: str, transform=None):
        self.transform = transform

        self.project = sly.Project(project_dir, sly.OpenMode.READ)
        self.meta = self.project.meta
        self.dataset = self.project.datasets.get(dataset)

        self.img_dir = self.dataset.img_dir
        self.label_dir = self.dataset.ann_dir

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_path = list(imutils.paths.list_images(self.img_dir))[idx]

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


if __name__ == "__main__":
    transform = A.Compose([
        A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.pytorch.transforms.ToTensorV2()
    ])
    dataset = AutofocusDataset(
        project_dir=r"C:\Users\tristan_cotte\PycharmProjects\prior_controller\autofocus\sly_project",
        dataset="ds0", transform=transform)
    print(dataset[2])
