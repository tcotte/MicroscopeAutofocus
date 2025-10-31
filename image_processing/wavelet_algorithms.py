import numpy as np
import pywt
import cv2
import exif
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import imutils
from imutils.paths import list_images


class WaveletAlgorithm:
    def __init__(self, filter: str = 'bior1.3'):
        self.wavelet_algorithm = filter

    def wavelet_decomposition(self, image):
        return pywt.dwt2(image, self.wavelet_algorithm)


class WaveletAlgo1(WaveletAlgorithm):
    def __init__(self, filter: str = 'bior1.3'):
        super().__init__(filter)

    def __call__(self, image, *args, **kwargs):
        LL, (LH, HL, HH) = self.wavelet_decomposition(image)
        return np.sum(np.abs(LH) + np.abs(HL) + np.abs(HH))


class WaveletAlgo3(WaveletAlgorithm):
    def __init__(self, filter: str = 'bior1.3'):
        super().__init__(filter)

    def __call__(self, image, *args, **kwargs):
        height, width = image.shape[:2]

        LL, (LH, HL, HH) = self.wavelet_decomposition(image)

        return np.sum((HL - np.mean(HL)) ** 2 + (LH - np.mean(LH)) ** 2 + (HH - np.mean(HH)) ** 2) / (height * width)


if __name__ == '__main__':

    image_folder = r'C:\Users\tristan_cotte\PycharmProjects\microscope_autofocus\autofocus\data\dataset_09_25_2025\X\test\76309x_39360y'

    wavelet_algo = WaveletAlgo3(filter='db6')

    list_g = []
    list_gt = []
    for image_path in natsorted(list(list_images(image_folder))):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        g = wavelet_algo(image=image)

        exif_metadata = exif.Image(image_path)
        list_gt.append(np.abs(float(exif_metadata.make)))

        list_g.append(g)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
    ax[0].plot(list_g)
    ax[0].set_title('Sharpness detection with Wavelet Algorithm')
    ax[0].set_xlabel('Z position (µm)')
    ax[0].set_ylabel('Sharpness value (UA)')
    ax[1].plot(list_gt)
    plt.show()
