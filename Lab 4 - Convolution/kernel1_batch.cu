#define STB_IMAGE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include "./include/dirent.h"
#include "./include/stb/stb_image_write.h"
#include "read_data.h"

__constant__ float constant_mask[MAX_MASK_SIZE * MAX_MASK_SIZE]; // constant memory for the mask

// Carry out a 3D convolution over RGB images and save the output ones
// 1. kernel1: basic implementation (no tiling) => each thread computes a single pixel in the output image
// to compute a single pixel in the output image the thread needs to compute values for all three channels using the corresponding mask for each channel
// each pixel reads three consecutive values from the input image
__global__ void kernel1_batch(unsigned char *output_images, float *input_images, int width, int height, int comp, int mask_size, int batch_size)
{
    // get the pixel index
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    // printf("column = %d, row = %d\n", column, row);

    // check if the pixel is within the image boundaries
    if (column < width && row < height && depth < batch_size)
    {
        float pixel_value = 0;

        // get the start of column and row for the current pixel being convulted
        int start_column = column - mask_size / 2;
        int start_row = row - mask_size / 2;
        // iterate over the three channels => for a single thread compute the pixel values for all three channels
        for (int c = 0; c < comp; c++)
        {
            // iterate over the mask elements => surrounding box
            for (int mask_row = 0; mask_row < mask_size; mask_row++) // rows
            {
                for (int mask_column = 0; mask_column < mask_size; mask_column++) // columns
                {
                    int current_row = start_row + mask_row;
                    int current_column = start_column + mask_column;

                    // check if the current pixel is within the image boundaries => no padding was added
                    if (current_row >= 0 && current_row < height && current_column >= 0 && current_column < width)
                    {
                        pixel_value += input_images[(width * height * depth + current_row * width + current_column) * comp + c] * constant_mask[mask_row * mask_size + mask_column];
                    }
                }
            }
        }

        pixel_value = fminf(fmaxf(pixel_value, 0.0f), 1.0f); // clamp the pixel value to be in the range [0, 1]

        pixel_value = pixel_value * 255; // scale the pixel value to be in the range [0, 255]

        // write the pixel value to the output image
        output_images[width * height * depth + row * width + column] = (unsigned char)pixel_value;
    }
}

// Dealing with the output images
// Deals with the threads, block and grid dimensions
// Calls the kernel to calculate the output images
// Transfers the output images from device to host
// Saves the output images
void calculateOutput(int depth, unsigned char *output_images, unsigned char *device_outputs, float *device_images, int mask_size, char *output_folder_path, char **output_image_filenames)
{
    // calculate the block and grid size
    dim3 block_dim(16, 16);
    int grid_columns = ceil((float)IMAGE_WIDTH / block_dim.x);
    int grid_rows = ceil((float)IMAGE_HEIGHT / block_dim.y);
    dim3 grid_dim(grid_columns, grid_rows, depth);

    // call the kernel on the batch of images read
    kernel1_batch<<<grid_dim, block_dim>>>(device_outputs, device_images, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, mask_size, depth);

    // transfer the output images from device to host
    cudaMemcpy(output_images, device_outputs, IMAGE_WIDTH * IMAGE_HEIGHT * depth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    printf("OUTPUT IMAGES COPIED TO HOST\n");

    // Get full output path
    char full_output_path[256];

    // save images
    for (int i = 0; i < depth; i++)
    {
        sprintf(full_output_path, "%s/%s", output_folder_path, output_image_filenames[i]);
        printf("FULL OUTPUT PATH: %s\n", full_output_path);

        stbi_write_jpg(full_output_path, IMAGE_WIDTH, IMAGE_HEIGHT, 1, output_images + i * IMAGE_WIDTH * IMAGE_HEIGHT, 100);
    }
}

int main(char argc, char *argv[])
{
    // Read and check command line arguments
    char *input_folder_path;
    char *output_folder_path;
    char *mask_file_path;

    int batch_size = readCommandLineArguments(argc, argv, &input_folder_path, &output_folder_path, &mask_file_path);

    // Allocate memory for filenames to save output images
    char **output_image_filenames = (char **)malloc(batch_size * sizeof(char *)); // Dynamic allocation for image names

    printf("%s\n", input_folder_path);
    printf("%s\n", output_folder_path);

    // 1. ALlocate host and device memory based on the batch size

    // 1.1 Host memory
    unsigned char *output_images = (unsigned char *)malloc(batch_size * IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(unsigned char));

    // 1.2 Device memory
    float *device_images;
    unsigned char *device_outputs;

    cudaMalloc((void **)&device_images, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS * batch_size);
    cudaMalloc((void **)&device_outputs, sizeof(unsigned char) * IMAGE_WIDTH * IMAGE_HEIGHT * batch_size);

    // 2. Read the mask and copy it to the contstant memory
    FILE *mask_file = fopen(mask_file_path, "r");
    if (!mask_file)
    {
        printf("Error: failed to read mask\n");
        exit(1);
    }

    // Read mask from file => same mask applied to all channels

    int mask_size = readMaskSize(mask_file);                              // read mask size
    float *mask = (float *)malloc(mask_size * mask_size * sizeof(float)); // allocate memory for the mask
    readMask(mask_file, mask, mask_size);                                 // read mask elements
    printMask(mask, mask_size);                                           // print mask elements

    // Copy Filter to constant memory
    cudaMemcpyToSymbol(constant_mask, mask, mask_size * mask_size * sizeof(float));

    // 3. Read images as batches from the input folder
    DIR *input_directory;
    struct dirent *entry;
    int batch_count = 0;

    if ((input_directory = opendir(input_folder_path)) != NULL)
    {
        printf("Input directory opened\n");

        // Read all images in batches and send to the GPU for convolution
        while ((entry = readdir(input_directory)) != NULL)
        {
            if (entry->d_type == DT_REG)
            {
                const char *image_name = entry->d_name;
                printf("Image File name = %s\n", image_name);
                output_image_filenames[batch_count] = strdup(image_name);

                // Get full input path
                char full_input_path[256];
                sprintf(full_input_path, "%s/%s", input_folder_path, image_name);
                printf("FULL INPUT PATH: %s\n", full_input_path);

                // Read the image
                int width, height, channels;
                unsigned char *input_image = readImage(full_input_path, &width, &height, &channels);

                if (input_image == NULL)
                {
                    printf("Error: failed to read image\n");
                    exit(1);
                }

                printf("width = %d, height = %d, channels = %d\n", width, height, channels);

                // normalize image
                float *normalized_image = (float *)malloc(width * height * channels * sizeof(float));
                for (size_t i = 0; i < width * height * channels; i++)
                {
                    normalized_image[i] = (float)input_image[i] / 255.0f;
                }

                assert(width == IMAGE_WIDTH && height == IMAGE_HEIGHT && channels == CHANNELS);

                // Copy image data from host to device
                cudaMemcpy(device_images + batch_count * width * height * channels, normalized_image, width * height * channels * sizeof(float), cudaMemcpyHostToDevice);

                printf("IMAGE COPIED TO GPU\n");

                free(input_image);
                free(normalized_image);

                batch_count++;
                if (batch_count >= batch_size)
                {
                    printf("BATCH COUNT: %d\n", batch_count);
                    calculateOutput(batch_size, output_images, device_outputs, device_images, mask_size, output_folder_path, output_image_filenames);
                    batch_count = 0;
                }
            }
        }

        printf("BATCH COUNT: %d\n", batch_count);
        if (batch_count != 0) // if the file_count % batch_size != 0
        {
            calculateOutput(batch_count, output_images, device_outputs, device_images, mask_size, output_folder_path, output_image_filenames);
        }

        // free dynamically allocated host memory
        free(mask);
        free(output_images);
        free(output_image_filenames);

        // free device memory
        cudaFree(device_images);
        cudaFree(device_outputs);

        // close the opened folders and files
        fclose(mask_file);
        closedir(input_directory);
    }
    else
    {
        // Error opening input directory
        perror("");
        return EXIT_FAILURE;
    }

    return 0;
}