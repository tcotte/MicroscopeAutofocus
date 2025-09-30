import os
from natsort import natsorted
import exif
import matplotlib.pyplot as plt
from imutils.paths import list_images

if __name__ == '__main__':
    slide_folder = r'D:\03 - IDEA\Micronoyaux\Autofocus\test\slide2'

    for folder in os.listdir(slide_folder):

        input_folder = os.path.join(slide_folder, folder)
        images = natsorted(list(list_images(input_folder)))

        z_positions = []
        for image in images:
            z_position = exif.Image(image).make
            z_positions.append(z_position)

        fig, ax = plt.subplots()
        ax.scatter(z_positions, list(range(len(z_positions))))
        ax.set_yticks(ax.get_yticks()[::2])
        ax.set_xticks(ax.get_xticks()[::20])
        plt.title(folder)
        plt.show()
