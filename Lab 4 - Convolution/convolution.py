# IMPORTS
import sys
import os
import math
import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


def read_images(input_folder_path):
    images = []
    images_names = []
    for image_name in os.listdir(input_folder_path):
        image = cv2.imread(input_folder_path + '/' + image_name)
        if image is not None:
            images.append(image)
            images_names.append(image_name)

    return images, images_names


def save_images(output_folder, images, images_names):
    for image, image_name in zip(images, images_names):
        print(image_name)
        cv2.imwrite(output_folder + '/' + image_name, image)


def read_mask(mask_file_path):
    with open(mask_file_path) as mask_file:
        mask_size = int(next(mask_file))
        mask = []
        for line in mask_file:
            mask.append([float(x) for x in line.split()])
        return mask_size, mask


def convolution(images, mask, mask_size, batch_size=1):
    padding = math.floor(mask_size / 2)

    # Define the convolutional layer outside of loop layer
    conv_layer = nn.Conv2d(in_channels=3, out_channels=3,
                           kernel_size=mask_size, padding=padding, bias=False)
    # Define weights for the kernel by our mask
    mask_tensor = torch.tensor(
        mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.cat([mask_tensor] * 3)

    # Set the weights of the convolutional layer
    with torch.no_grad():  # Disable gradient tracking to avoid unnecessary computation
        conv_layer.weight.copy_(mask_tensor)

    # Divide images into batches
    out_images = []
    for i in range(0, len(images), batch_size):
        if i+batch_size > len(images):
            batch_images = images[i:]
        else:
            batch_images = images[i:i+batch_size]

        # Stack images along the batch dimension
        input_batch_images = torch.stack(
            [torch.tensor(image) for image in batch_images])

        # Normalize the pixel values to the range [0, 1] and adjust the shape
        # Assuming the input images are in the range [0, 255]
        input_batch_images = input_batch_images.permute(
            0, 3, 1, 2).float() / 255

        # Apply convolutional layer to the batch of images
        output_tensor = conv_layer(input_batch_images)

        # Convert output tensor back to PIL images
        batch_outputs = [np.array(T.ToPILImage()(img.squeeze(0)))
                         for img in output_tensor]
        out_images.extend(batch_outputs)
    return out_images


def main():

    # Check if correct number of arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python <input path> <output path> <batch size> <mask path>")
        return

    # Save command line arguments
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    batch_size = int(sys.argv[3])
    mask_file = sys.argv[4]

    images, names = read_images(input_folder)
    mask_size, mask = read_mask(mask_file)
    out_images = convolution(images, mask, mask_size, batch_size)
    save_images(output_folder , out_images, names)


if __name__ == "__main__":
    main()
