import sys
import os
import cv2


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


def convolution():
    pass


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


if __name__ == "__main__":
    main()
