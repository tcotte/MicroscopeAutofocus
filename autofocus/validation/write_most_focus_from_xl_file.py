import os.path
import shutil

import pandas as pd
from imutils.paths import list_images

if __name__ == '__main__':
    df = pd.read_excel(r'D:\03 - IDEA\Micronoyaux\Autofocus\datasets_v2\y\slide_4_annotation_file.xlsx')

    image_list = list(list_images(r'D:\03 - IDEA\Micronoyaux\Autofocus\datasets_v2\X\slide_4'))

    for tag in df['optimal_Z']:
        x_pos, y_pos, z_pos = tag.split('_')
        img_name = f'{x_pos}_{y_pos}_{z_pos}.jpg'

        not_found = True
        for image_path in image_list:
            if os.path.basename(image_path) == img_name:
                shutil.copy(image_path, fr'C:\Users\tristan_cotte\PycharmProjects\microscope_autofocus\output_validation\most_focus_pictures\slide4_xl\{img_name}')
                not_found = False

        if not_found:
            print(img_name)