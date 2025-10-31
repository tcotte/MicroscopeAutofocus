import cv2
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import imutils
from imutils.paths import list_images
from tqdm import tqdm


class StdBasedCorrelation(object):
    def __init__(self):
        pass

    def __call__(self, gray_image, *args, **kwargs):
        height, width = gray_image.shape[:2]
        mean = np.mean(gray_image)

        std_based_correlation = 0

        for y in range(gray_image.shape[0]):
            for x in range(gray_image.shape[1] - 1):
                p0 = gray_image[y, x]
                p1 = gray_image[y, + 1]

                std_based_correlation += p0 * p1 - height * width * mean ** 2

        return std_based_correlation


class FasterStdBasedCorrelation(object):
    def __init__(self, normalize: bool):
        """
        Some SDC variants normalize the correlation by the image variance.
        To compare between images with different brightness/contrast, normalization helps.
        :param normalize: boolean which indicates if we use normalization or not.
        Normalization is recommended because without the algorithm is not scale-invariant. That means:
        - If two images have the same sharpness, but different brightness or contrast, the raw correlation CC will still
        change.
        - This makes the focus score sensitive to changes in lighting, gain, or sensor variability — not just focus.
        However, when the correlation is divided by image variance, it measures how correlated neighboring pixels are,
        relative to the overall intensity variation in the image. This creates a dimensionless, scale-invariant metric
        that isolates structural correlation (what is sharpness about).

        """
        self._normalize = normalize

    def __call__(self, gray_image, *args, **kwargs):
        height, width = gray_image.shape[:2]

        # Compute the mean of the image
        mean = np.mean(gray_image)

        # Precompute the constant value to be subtracted in the correlation
        mean_squared_term = height * width * mean ** 2

        # Calculate the correlation between consecutive pixels
        # Use np.roll to shift the image by 1 pixel and calculate pairwise products
        shifted_image = np.roll(gray_image, -1, axis=1)
        shifted_image[:, -1] = 0  # zero out wrapped values
        correlation_matrix = gray_image * shifted_image

        # Sum up all the correlation values and subtract the mean squared term
        std_based_correlation = np.sum(correlation_matrix) - mean_squared_term

        if self._normalize:
            std = np.std(gray_image)
            if std > 0:
                std_based_correlation /= std ** 2

        return std_based_correlation


class Variance:
    def __init__(self, normalize=True):
        self._normalize = normalize

    def __call__(self, gray_image, *args, **kwargs):
        height, width = gray_image.shape[:2]

        if not self._normalize:
            return np.sum(gray_image - np.mean(gray_image)**2) / (height * width)

        else:
            std = np.std(gray_image)
            if std > 0:
                return np.sum(gray_image - np.mean(gray_image)**2) / (height * width) / std ** 2

            else:
                return np.sum(gray_image - np.mean(gray_image)**2) / (height * width)

if __name__ == '__main__':

    image_folder = r'C:\Users\tristan_cotte\PycharmProjects\microscope_autofocus\autofocus\data\dataset_09_25_2025\X\train\71224x_16350y'

    statistics_based_algo = Variance(normalize=True)

    list_focus_value = []
    for image_path in tqdm(natsorted(list(list_images(image_folder)))):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        value = statistics_based_algo(gray_image=image)

        list_focus_value.append(value)

    plt.plot(list_focus_value)
    plt.show()
