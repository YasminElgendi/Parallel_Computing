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

#define OUTPUT_TILE_WIDTH 16 // => 16 x 16 = 256

__constant__ float constant_mask[MAX_MASK_SIZE * MAX_MASK_SIZE]; // constant memory for the mask

// 3. kernel3: tiling where each block matches the output tile size.
// The size of the block matches the size of the input tile
__global__ void kernel3_batch(unsigned char *output_images, float *input_images, int width, int height, int comp, int mask_size, int batch_size, int input_tile_width)
{
    // get the pixel index in the output image
    int out_column = blockIdx.x * OUTPUT_TILE_WIDTH + threadIdx.x;
    int out_row = blockIdx.y * OUTPUT_TILE_WIDTH + threadIdx.y;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    // indices to access the shared memory
    int in_column = out_column - mask_size / 2;
    int in_row = out_row - mask_size / 2;

    // Indices that are used to access the input image
    int start_block_column = blockIdx.x * OUTPUT_TILE_WIDTH - mask_size / 2;
    int start_block_row = blockIdx.y * OUTPUT_TILE_WIDTH - mask_size / 2;

    // the number of elements that each thread should load
    int elements_per_thread = ((input_tile_width * input_tile_width) / (OUTPUT_TILE_WIDTH * OUTPUT_TILE_WIDTH)) + 1;

    // STEPS:
    // 1. Load data into shared memory
    // some threads will load more than one pixel to the shared memeory
    // each thread will load 3 values corresponding to the three channels
    extern __shared__ float shared_input[]; // Example: Allocate the input tile size of shared memory

    if (depth < batch_size)
    {
        int stride = OUTPUT_TILE_WIDTH * OUTPUT_TILE_WIDTH;

        for (int i = 0; i < elements_per_thread; i++) // for each thread iterate ove the elements that it should load => same element in eaxg row in the inpyt tile
        {
            // the thread index with respect to the rest of the threads
            // if the output tile is 3x3 => 9 threads
            // the thread_index should go from 0-8
            int thread_index = threadIdx.y * OUTPUT_TILE_WIDTH + threadIdx.x;

            // since a single thread loads more than one input element
            // the step is the difference between elements for which a single thread loads its in
            int thread_index_step = thread_index + (i * stride);

            // get the indices of the the thread with respect to the input tile
            // if a thread is number 8 and the input tile is 5x5 => then the thread loads into the cell 1,3 in the shared memory
            int shm_index_row = thread_index_step / input_tile_width;
            int shm_index_col = thread_index_step - (shm_index_row * input_tile_width);

            if (shm_index_col >= 0 && shm_index_col < input_tile_width && shm_index_row >= 0 && shm_index_row < input_tile_width)
            {

                // get the index of the thread with respect to the input image
                // use the in_column and in_row => wrong
                // use the shared memory indices
                int input_index_col = start_block_column + shm_index_col;
                int input_index_row = start_block_row + shm_index_row;

                if (input_index_col >= 0 && input_index_col < width && input_index_row >= 0 && input_index_row < height)
                {
                    for (int c = 0; c < comp; c++) // this does not support memory coalescing
                    {
                        // Load the pixel value from the input image into the shared memory if the image is in bounds
                        shared_input[(shm_index_row * input_tile_width + shm_index_col) * comp + c] = input_images[(width * height * depth + input_index_row * width + input_index_col) * comp + c];
                    }
                }
                else
                {
                    // this does not support memory coalescing
                    // since the image is saved to memory where each pixel is saved in three consecutive cells
                    // we want the threads to load consecutive cells
                    // if I have 3 threads and three pixels => thread 0 loads first channel of each pixel, thread 1 loads the second channel of each pixel, thread 2 loads the third channel of each pixel => ezay ba2a
                    for (int c = 0; c < comp; c++)
                    {
                        // Insert a 0 if the index is out of bounds => ghost cells
                        shared_input[(shm_index_row * input_tile_width + shm_index_col) * comp + c] = 0.0f;
                    }
                }
            }
        }
    }

    __syncthreads();

    // 2. Compute the output tile

    // all threads will participate in computing the pixel value
    // no need to check if the thread is within the boundaries
    if (out_column < width && out_row < height && depth < batch_size)
    {
        float pixel_value = 0.0f;
        for (int c = 0; c < comp; c++)
        {
            // iterate over the mask elements => surrounding box
            for (int mask_row = 0; mask_row < mask_size; mask_row++) // rows
            {
                for (int mask_column = 0; mask_column < mask_size; mask_column++) // columns
                {
                    pixel_value += shared_input[((threadIdx.y + mask_row) * input_tile_width + (threadIdx.x + mask_column)) * comp + c] * constant_mask[mask_row * mask_size + mask_column];
                }
            }
        }

        pixel_value = fminf(fmaxf(pixel_value, 0.0f), 1.0f); // clamp the pixel value to be in the range [0, 1]

        pixel_value = pixel_value * 255; // scale the pixel value to be in the range [0, 255]

        // 3. Write the output tile to the output image
        output_images[width * height * depth + out_row * width + out_column] = (unsigned char)pixel_value;
    }
}

// Dealing with the output images
// Deals with the threads, block and grid dimensions
// Calls the kernel to calculate the output images
// Transfers the output images from device to host
// Saves the output images
void calculateOutput(int width, int height, int channels, int depth, unsigned char *output_images, unsigned char *device_outputs, float *device_images, int mask_size, char *output_folder_path, char **output_image_filenames)
{
    // calculate the block and grid size

    int input_tile_width = OUTPUT_TILE_WIDTH + mask_size - 1;

    dim3 block_dim(OUTPUT_TILE_WIDTH, OUTPUT_TILE_WIDTH);
    int grid_columns = ceil((float)width / OUTPUT_TILE_WIDTH);
    int grid_rows = ceil((float)height / OUTPUT_TILE_WIDTH);
    dim3 grid_dim(grid_columns, grid_rows, depth);

    int shared_memory_size = input_tile_width * input_tile_width * channels * sizeof(float);

    // call the kernel on the batch of images read
    kernel3_batch<<<grid_dim, block_dim, shared_memory_size>>>(device_outputs, device_images, width, height, channels, mask_size, depth, input_tile_width);

    // transfer the output images from device to host
    cudaMemcpy(output_images, device_outputs, width * height * depth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    printf("OUTPUT IMAGES COPIED TO HOST\n");

    // Get full output path
    char full_output_path[256];

    // save images
    for (int i = 0; i < depth; i++)
    {
        sprintf(full_output_path, "%s/%s", output_folder_path, output_image_filenames[i]);
        printf("FULL OUTPUT PATH: %s\n", full_output_path);

        stbi_write_jpg(full_output_path, width, height, 1, output_images + i * width * height, 100);
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

    // First get the dimension of the images => all images are supposed to have the same dimension => get the dimension of the first image
    int WIDTH, HEIGHT, CHANNELS; // image dimensions for all images in the input folder
    bool success = getImageDimensions(input_folder_path, &WIDTH, &HEIGHT, &CHANNELS);

    if (!success)
    {
        printf("Error: failed to read image dimensions\n");
        exit(1);
    }

    // 1. ALlocate host and device memory based on the batch size

    // 1.1 Host memory
    unsigned char *output_images = (unsigned char *)malloc(batch_size * WIDTH * HEIGHT * sizeof(unsigned char));

    // 1.2 Device memory
    float *device_images;
    unsigned char *device_outputs;

    cudaMalloc((void **)&device_images, sizeof(float) * WIDTH * HEIGHT * CHANNELS * batch_size);
    cudaMalloc((void **)&device_outputs, sizeof(unsigned char) * WIDTH * HEIGHT * batch_size);

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
                printf("width = %d, height = %d, channels = %d\n", width, height, channels);
                if (input_image == NULL)
                {
                    printf("Error: failed to read image\n");
                    exit(1);
                }

                assert(width == WIDTH && height == HEIGHT && channels == CHANNELS);

                // normalize image => convert to float between 0 and 1
                float *normalized_image = (float *)malloc(WIDTH * HEIGHT * CHANNELS * sizeof(float));
                for (size_t i = 0; i < WIDTH * HEIGHT * CHANNELS; i++)
                {
                    normalized_image[i] = (float)input_image[i] / 255.0f;
                }

                // Copy image data from host to device
                cudaMemcpy(device_images + batch_count * WIDTH * HEIGHT * CHANNELS, normalized_image, WIDTH * HEIGHT * CHANNELS * sizeof(float), cudaMemcpyHostToDevice);

                printf("IMAGE COPIED TO GPU\n");

                free(input_image);
                free(normalized_image);

                batch_count++;
                if (batch_count >= batch_size)
                {
                    printf("BATCH COUNT: %d\n", batch_count);
                    calculateOutput(WIDTH, HEIGHT, CHANNELS, batch_size, output_images, device_outputs, device_images, mask_size, output_folder_path, output_image_filenames);
                    batch_count = 0;
                }
            }
        }

        printf("BATCH COUNT: %d\n", batch_count);
        if (batch_count != 0) // if the file_count % batch_size != 0
        {
            calculateOutput(WIDTH, HEIGHT, CHANNELS, batch_count, output_images, device_outputs, device_images, mask_size, output_folder_path, output_image_filenames);
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