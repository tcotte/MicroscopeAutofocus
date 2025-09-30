import os

import cv2
from natsort import natsorted
import exif
import matplotlib.pyplot as plt
from imutils.paths import list_images

if __name__ == '__main__':
    slide_folder = r'C:\Users\tristan_cotte\PycharmProjects\microscope_autofocus\autofocus\data\datasets_v2 - Copy (2)\X\slide_4'

    for folder in os.listdir(slide_folder):

        input_folder = os.path.join(slide_folder, folder)
        images = natsorted(list(list_images(input_folder)))

        for image in images:
            z_position = exif.Image(image).make
            if float(z_position) == 0:
                print(folder)
                cv2.imwrite(filename=os.path.join(r'C:\Users\tristan_cotte\PycharmProjects\microscope_autofocus\output_validation\most_focus_pictures\slide4', os.path.basename(image)),
                            img=cv2.imread(image))

        # fig, ax = plt.subplots()
        # ax.scatter(z_positions, list(range(len(z_positions))))
        # ax.set_yticks(ax.get_yticks()[::2])
        # ax.set_xticks(ax.get_xticks()[::20])
        # plt.title(folder)
        # plt.show()