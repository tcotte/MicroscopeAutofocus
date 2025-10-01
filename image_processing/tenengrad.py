import cv2
import exif
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import imutils
from imutils.paths import list_images


class Tenengrad:
    def __init__(self):
        self.sobel_x = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]])

        self.sobel_y = np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])

    # Define the convolution function
    def convolution2d(self, input_matrix, kernel):
        # Get dimensions of input and kernel
        input_h, input_w = input_matrix.shape
        kernel_h, kernel_w = kernel.shape

        # Calculate the size of the output matrix
        output_h = input_h - kernel_h + 1
        output_w = input_w - kernel_w + 1

        # Create an empty output matrix
        output = np.zeros((output_h, output_w))

        # Perform the convolution operation
        for i in range(output_h):
            for j in range(output_w):
                # Extract the region of the input matrix covered by the kernel
                region = input_matrix[i:i + kernel_h, j:j + kernel_w]

                # Apply element-wise multiplication and sum the result
                output[i, j] = np.sum(region * kernel)

        return output

    def __call__(self, image, *args, **kwargs):
        # Perform convolution

        g_x = self.convolution2d(image, kernel=self.sobel_x)
        g_y = self.convolution2d(image, kernel=self.sobel_x)
        g = np.sqrt(np.square(g_x) + np.square(g_y))
        return np.sum(g)


class FasterTenengrad:
    def __call__(self, image, *args, **kwargs):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
        g = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return np.sum(g)


if __name__ == '__main__':

    image_folder = r'C:\Users\tristan_cotte\PycharmProjects\microscope_autofocus\autofocus\data\dataset_09_25_2025\X\train\79479x_38942y'

    tenengrad = FasterTenengrad()

    list_g = []
    list_gt = []
    for image_path in natsorted(list(list_images(image_folder))):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        g = tenengrad(image=image)

        exif_metadata = exif.Image(image_path)
        list_gt.append(np.abs(float(exif_metadata.make)))

        list_g.append(g)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
    ax[0].plot(list_g)
    ax[1].plot(list_gt)
    plt.show()
