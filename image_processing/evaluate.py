import os

import pandas as pd

from image_processing.std_based_correlation import FasterStdBasedCorrelation
from image_processing.tenengrad import FasterTenengrad

import cv2
import exif
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from imutils.paths import list_images

from image_processing.wavelet_algorithms import WaveletAlgo3


class Normalizer:
    def __call__(self, list_values, *args, **kwargs):
        return [float(i) / max(list_values) for i in list_values]


class Metrics:

    def accuracy(self, y, y_hat):
        return np.abs(y - y_hat)


def get_z_position_from_index(list_sorted_images, index):
    image_path = list_sorted_images[index]
    image_name = os.path.basename(image_path)
    str_z_position = image_name.split('_')[-1][:-4].replace(',', '.')
    return float(str_z_position)


if __name__ == '__main__':

    training_set = (r'C:\Users\tristan_cotte\PycharmProjects\microscope_autofocus\autofocus\data\dataset_09_25_2025\X'
                    r'\train')

    plot: bool = False

    tenengrad = FasterTenengrad()
    wavelet = WaveletAlgo3(filter='db6')
    std_based_correlation = FasterStdBasedCorrelation(normalize=True)

    algorithms = {'tenengrad': FasterTenengrad(),
                  'wavelet': WaveletAlgo3(filter='db6'),
                  'std_based_correlation': FasterStdBasedCorrelation(normalize=True)}

    data = {'xy_positions': [],
            'ground_truth': []}

    for i in list(algorithms.keys()):
        data[f'accuracy_{i}'] = []

    dict_accuracies = {}

    metrics = Metrics()

    normalizer = Normalizer()

    for image_folder in os.listdir(training_set):



        xy_position = os.path.join(training_set, image_folder)

        for key, algo in algorithms.items():
            list_g = []
            list_gt = []

            for image_path in natsorted(list(list_images(xy_position))):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                g = algo(image)

                exif_metadata = exif.Image(image_path)
                list_gt.append(np.abs(float(exif_metadata.make)))

                list_g.append(g)

            y_hat = get_z_position_from_index(list_sorted_images=natsorted(list(list_images(xy_position))),
                                              index=int(np.argmax(list_g)))
            y = get_z_position_from_index(list_sorted_images=natsorted(list(list_images(xy_position))),
                                          index=list_gt.index(0))

            if image_folder == '99377x_10214y':
                print(f'{key} - y_hat={y_hat} - y={y}')

            accuracy = metrics.accuracy(y=y, y_hat=y_hat)

            dict_accuracies[key] = accuracy

            if plot:
                plt.title(f'Accuracy: {accuracy:.2f} on {image_folder} with algorithm {key}')
                plt.plot(normalizer(list_g), label='Focus algorithm (norm.')
                plt.plot(normalizer(list_gt), label='Absolute distance_af from focus (norm.)')
                plt.legend()
                plt.show()

        data['xy_positions'].append(image_folder)
        data['ground_truth'].append(y_hat)
        for key in dict_accuracies.keys():
            data[f'accuracy_{key}'].append(dict_accuracies[key])

        # fig, ax = plt.plot(figsize=(15, 8))


    df = pd.DataFrame(data=data)
    df.to_excel('data.xlsx', index=False)
