import cv2
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Linear, Dropout, Hardswish
from torchvision.transforms import transforms

model = torchvision.models.mobilenet_v3_small()

def rename_attribute(obj, old_name, new_name):
    obj._modules[new_name] = obj._modules.pop(old_name)


# class Mobilnetv3Regressor(torchvision.models.mobilenet_v3_small):
#     def __init__(self):
#         self.classifier = nn.Sequential(Linear(in_features=576, out_features=1024),
#                                     Hardswish(),
#                                     Dropout(p=0.2),
#                                     Linear(in_features=1024, out_features=1))


if __name__ == "__main__":
    image = Image.open(r"C:\Users\tristan_cotte\PycharmProjects\prior_controller\autofocus\sly_project\ds0\img\1976_1976_10.jpg")

    # Define a transform to convert the image to tensor
    transform = transforms.ToTensor()

    # Convert the image to PyTorch tensor
    tensor = transform(image)
    tensor = torch.unsqueeze(tensor, dim=0)
    print(tensor.shape)
    # print(model)
    #     #
    #     # model.features[-1] = nn.Linear(in_features=576, out_features=1)
    #     # print(model)

    # for n, m in model.named_modules():
    #     print(n, m)
    #
    # print(model.classifier)
    #
    # print(model.__getattr__('classifier'))

    model.classifier = nn.Sequential(Linear(in_features=576, out_features=1024),
                                        Hardswish(),
                                        Dropout(p=0.2),
                                        Linear(in_features=1024, out_features=1))
    # rename_attribute(model, 'classifier', 'regressor')
    print(model)
    with torch.no_grad():
        print(model(tensor))
