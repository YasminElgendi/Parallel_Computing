import cv2
import numpy as np
import os


def compare_images(kernel1_images, kernel2_images, kernel3_images):

    # # Convert images to grayscale
    # gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute Mean Squared Error (MSE) for all images

    mse = (0, 0, 0)  # 0 => mse 1,2 | 1 => mse 1,3 | 2 => mse 2,3

    for img1, img2, img3 in zip(kernel1_images, kernel2_images, kernel3_images):
        mse_1_2 = np.mean((img1 - img2) ** 2)
        mse_1_3 = np.mean((img1 - img3) ** 2)
        mse_2_3 = np.mean((img2 - img3) ** 2)

        [sum(x) for x in zip(mse, (mse_1_2, mse_1_3, mse_2_3))]

    return mse


def read_images(kernel1_path, kernel2_path, kernel3_path):

    # Read the images
    kernel1_images = []
    kernel2_images = []
    kernel3_images = []

    content1 = os.listdir(kernel1_path)
    content2 = os.listdir(kernel2_path)
    content3 = os.listdir(kernel3_path)

    assert len(content1) == len(content2) == len(
        content3), "Number of images in the folders are not the same"

    for image1, image2, image3 in zip(content1, content2, content3):
        img = cv2.imread(os.path.join(kernel1_path, image1),
                         cv2.IMREAD_GRAYSCALE)
        kernel1_images.append(img)

        img = cv2.imread(os.path.join(kernel2_path, image2),
                         cv2.IMREAD_GRAYSCALE)
        kernel2_images.append(img)

        img = cv2.imread(os.path.join(kernel3_path, image3),
                         cv2.IMREAD_GRAYSCALE)
        kernel3_images.append(img)

    return kernel1_images, kernel2_images, kernel3_images


if __name__ == "__main__":
    # Paths to the images
    kernel1_path = "./output/kernel1"
    kernel2_path = "./output/kernel2"
    kernel3_path = "./output/kernel3"

    kernel1_images, kernel2_images, kernel3_images = read_images(
        kernel1_path, kernel2_path, kernel3_path)

    # Compare images
    mse = compare_images(kernel1_images, kernel2_images, kernel3_images)

    # Print the results
    print(f"Mean Squared Error (1_2, 2_1, 1_3): {mse}")
    # print(f"Structural Similarity Index (SSIM): {ssim}")
