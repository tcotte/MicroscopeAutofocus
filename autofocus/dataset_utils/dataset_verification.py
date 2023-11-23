import json
import os

import cv2
import matplotlib.pyplot as plt

from autofocus.utils import get_os


def get_focus_diff_from_json(annotation_file):
    with open(annotation_file) as json_file:
        data = json.load(json_file)

    for i in data["tags"]:
        if i.get("name") == "focus_difference":
            return i.get("value")

    raise "Focus difference tags was not found"


def get_img_file_from_ann_file(annotation_path):
    path = os.path.normpath(annotation_path)
    splitted_path = path.split(os.sep)
    if get_os() == "Windows":
        img_path = os.path.join("C://", *splitted_path[1:-2], "img", splitted_path[-1][:-4])
    else:
        return None
    # else:
    #     label_path = os.path.join(*splitted_path[:-2], "ann", splitted_path[-1] + ".json")
    return img_path
path_dataset = r"C:\Users\tristan_cotte\PycharmProjects\prior_controller\output_picture\acquisition_first_slide"

if __name__ == "__main__":
    for directory_name in os.listdir(path_dataset):

        directory = os.path.join(path_dataset, directory_name)
        if os.path.isdir(directory):
            annotation_directory = os.path.join(directory, "ann")

            for json_file in os.listdir(annotation_directory):
                json_path = os.path.join(annotation_directory, json_file)

                z_value = get_focus_diff_from_json(annotation_file=json_path)

                if z_value == 0:
                    print(json_path)
                    img_path = get_img_file_from_ann_file(json_path)
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img_rgb)
                    plt.show()

