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
#include "./include/stb/stb_image_write.h"
#include "read_data.h"

__constant__ float constant_mask[MAX_MASK_SIZE * MAX_MASK_SIZE]; // constant memory for the mask

// Carry out a 3D convolution over RGB images and save the output ones
// 1. kernel1: basic implementation (no tiling) => each thread computes a single pixel in the output image
// to compute a single pixel in the output image the thread needs to compute values for all three channels using the corresponding mask for each channel
// each pixel reads three consecutive values from the input image
__global__ void kernel1(unsigned char *output_image, float *input_image, int width, int height, int comp, int mask_size)
{
    // get the pixel index
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("column = %d, row = %d\n", column, row);

    // check if the pixel is within the image boundaries
    if (column < width && row < height)
    {
        float pixel_value = 0.0f;

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
                        pixel_value += input_image[(current_row * width + current_column) * comp + c] * constant_mask[mask_row * mask_size + mask_column];
                    }
                }
            }
        }

        pixel_value = fminf(fmaxf(pixel_value, 0.0f), 1.0f); // clamp the pixel value to be in the range [0, 1]

        pixel_value = pixel_value * 255; // scale the pixel value to be in the range [0, 255]

        output_image[row * width + column] = (unsigned char)pixel_value; // write the pixel value to the output image
    }
}

int main(char argc, char *argv[])
{
    // Read and check command line arguments
    // Command line arguments
    char *input_folder_path;
    char *output_folder_path;
    char *mask_file_path;

    int batch_size = readCommandLineArguments(argc, argv, &input_folder_path, &output_folder_path, &mask_file_path);

    // 1. Allocate host memory => read input image / convolution mask

    // 1.1 Read image
    int width, height, comp;
    unsigned char *input_image = readImage(strcat(input_folder_path, "/tree.jpg"), &width, &height, &comp);
    if (!input_image)
    {
        printf("Error: failed to read image\n");
        exit(1);
    }

    printf("INPUT IMAGE\n");
    for (size_t i = 0; i < NUM_PIXELS_TO_PRINT * comp; i++)
    {
        printf("%d%s", input_image[i], ((i + 1) % comp) ? " " : "\n");
    }

    printf("FLOAT INPUT IMAGE\n");
    for (size_t i = 0; i < NUM_PIXELS_TO_PRINT * comp; i++)
    {
        printf("%f%s", (float)input_image[i] / 255.0, ((i + 1) % comp) ? " " : "\n");
    }

    // convert the image to float
    float *normalized_image = (float *)malloc(width * height * comp * sizeof(float));
    for (size_t i = 0; i < width * height * comp; i++)
    {
        normalized_image[i] = (float)input_image[i] / 255.0;
        // printf("%f%s", normalized_image[i], ((i + 1) % comp) ? " " : "\n");
    }

    printf("NORMALIZED IMAGE\n");
    for (size_t i = 0; i < NUM_PIXELS_TO_PRINT * comp; i++)
    {
        printf("%f%s", normalized_image[i], ((i + 1) % comp) ? " " : "\n");
    }

    printf("width = %d, height = %d, comp = %d (channels)\n", width, height, comp); // print image dimensions
    int N = width * height * comp;
    unsigned char *output_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    // 1.2 Read convolution masks
    FILE *mask_file = fopen(mask_file_path, "r");
    if (!mask_file)
    {
        printf("Error: failed to read mask\n");
        exit(1);
    }

    // Read mask from file => same mask applied on all channels

    // read the first mask for the first channel
    int mask_size = readMaskSize(mask_file);                              // read mask size
    float *mask = (float *)malloc(mask_size * mask_size * sizeof(float)); // allocate memory for the mask
    readMask(mask_file, mask, mask_size);                                 // read mask elements
    printMask(mask, mask_size);                                           // print mask elements

    // Copy Filter to constant memory
    cudaMemcpyToSymbol(constant_mask, mask, mask_size * mask_size * sizeof(float));

    // 2. Allocate device memory
    float *d_image;
    unsigned char *d_out;
    cudaMalloc((void **)&d_image, sizeof(float) * N);
    cudaMalloc((void **)&d_out, sizeof(unsigned char) * width * height);

    // 3. Transfer data from host to device memory
    cudaMemcpy(d_image, normalized_image, sizeof(float) * N, cudaMemcpyHostToDevice);

    // 4. Launch kernel
    dim3 block_dim(16, 16);
    int grid_columns = ceil((float)width / block_dim.x);
    int grid_rows = ceil((float)height / block_dim.y);
    dim3 grid_dim(grid_columns, grid_rows);

    kernel1<<<grid_dim, block_dim>>>(d_out, d_image, width, height, comp, mask_size);

    // If Error occurs in kernel execution show it using cudaDeviceSynchronize, cudaGetLastError
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // in red
        printf("\033[1;31m");
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        // reset color
        printf("\033[0m");
    }

    // 5. Transfer data from device to host memory
    cudaMemcpy(output_image, d_out, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

    // 6. Write output image

    stbi_write_jpg("./output/output_image.jpg", width, height, 1, output_image, 100);
    printf("OUTPUT IMAGE\n");
    for (size_t i = 0; i < NUM_PIXELS_TO_PRINT * comp; i++)
    {
        printf("%d%s", output_image[i], ((i + 1) % comp) ? " " : "\n");
    }

    // close the files
    fclose(mask_file);

    // free host memory
    free(mask);
    free(input_image);
    free(normalized_image);
    free(output_image);

    // free device memory
    cudaFree(d_image);
    cudaFree(d_out);
    cudaFree(constant_mask);

    return 0;
}