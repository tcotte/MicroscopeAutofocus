import os

import exif
import imutils.paths
import numpy as np
import pandas as pd
from natsort import index_natsorted, order_by_index
from tqdm import tqdm

gamma: float = 2
tolerance: float = 0.5

if __name__ == '__main__':

    train_dataset_folder = r'D:\03 - IDEA\Micronoyaux\Autofocus\dataset_11_13_2025\X\train'
    test_dataset_folder = r'D:\03 - IDEA\Micronoyaux\Autofocus\dataset_11_13_2025\X\test'

    for set_folder, set_name in zip([train_dataset_folder, test_dataset_folder], ['train', 'test']):
        df = pd.DataFrame()

        for xy_position in tqdm(os.listdir(set_folder)):
            folder_xy_position = os.path.join(set_folder, xy_position)

            list_distances = []
            list_images = []
            for image in list(imutils.paths.list_images(folder_xy_position)):
                distance_af = float(exif.Image(image).make)
                list_distances.append(round(distance_af, 2))
                list_images.append(image)
                print(f'{os.path.basename(image)} -> {distance_af:.2f}')

            indexes = index_natsorted(list_distances)
            list_distances = np.array(order_by_index(list_distances, indexes))
            list_images = order_by_index(list_images, indexes)

            pair_found = 0
            for distance_af in list_distances[list_distances < -gamma]:
                if distance_af + gamma in list_distances:
                    index_z1 = list_distances.tolist().index(distance_af)
                    index_z2 = list_distances.tolist().index(distance_af + gamma)

                    item_data = {'xy_position': [os.path.basename(folder_xy_position)],
                                 'z1_image': [os.path.basename(list_images[index_z1])],
                                 'z2_image': [os.path.basename(list_images[index_z2])],
                                 'z1_diff_focus': [distance_af],
                                 'z2_diff_focus': [distance_af + gamma]}
                    df = pd.concat([df, pd.DataFrame(data=item_data)], ignore_index=True)
                    print(df)


                    pair_found += 1

            for distance_af in list_distances[list_distances > 0]:
                if distance_af + gamma in list_distances:
                    index_z1 = list_distances.tolist().index(distance_af)
                    index_z2 = list_distances.tolist().index(distance_af + gamma)

                    item_data = {'xy_position': [os.path.basename(folder_xy_position)],
                                 'z1_image': [os.path.basename(list_images[index_z1])],
                                 'z2_image': [os.path.basename(list_images[index_z2])],
                                 'z1_diff_focus': [distance_af],
                                 'z2_diff_focus': [distance_af + gamma]}
                    df = pd.concat([df, pd.DataFrame(item_data)], ignore_index=True)

                    pair_found += 1

            print(f'pairs found: {pair_found} with a stack of {len(list_distances)} items')

        df.to_excel(f'diff_{set_name}.xlsx')
