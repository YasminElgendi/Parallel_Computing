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

#define OUTPUT_TILE_WIDTH 4 // => 16 x 16 = 256

__constant__ float constant_mask[MAX_MASK_SIZE * MAX_MASK_SIZE]; // constant memory for the mask

// 3. kernel3: tiling where each block matches the output tile size.
// The size of the block matches the size of the input tile
__global__ void kernel3(unsigned char *output_image, float *input_image, int width, int height, int comp, int mask_size, int input_tile_width)
{
    // get the pixel index
    int out_column = blockIdx.x * OUTPUT_TILE_WIDTH + threadIdx.x;
    int out_row = blockIdx.y * OUTPUT_TILE_WIDTH + threadIdx.y;

    // printf("blockIdx.x = %d, blockIdx.y = %d\n", blockIdx.x, blockIdx.y);
    // printf("threadIdx.x = %d, threadIdx.y = %d\n", threadIdx.x, threadIdx.y);

    // printf("out_column = %d, out_row = %d\n", out_column, out_row);

    // indices to access the shared memory
    // This only works if each thread is loading one pixel from the input image into thae shared memory
    // We need another way to access the input image
    // use for boundary conditions
    int in_column = out_column - mask_size / 2;
    int in_row = out_row - mask_size / 2;

    // Indices that are used to access the input image
    int start_block_column = blockIdx.x * OUTPUT_TILE_WIDTH - mask_size / 2;
    int start_block_row = blockIdx.y * OUTPUT_TILE_WIDTH - mask_size / 2;

    // printf("in_column = %d, in_row = %d\n", in_column, in_row);

    int elements_per_thread = ((input_tile_width * input_tile_width) / (OUTPUT_TILE_WIDTH * OUTPUT_TILE_WIDTH)) + 1;
    // printf("elements_per_thread = %d\n", elements_per_thread);

    // STEPS:
    // 1. Load data into shared memory
    // some threads will load more than one pixel to the shared memeory
    // each thread will load 3 values corresponding to the three channels
    extern __shared__ float shared_input[]; // Example: Allocate the input tile size of shared memory

    // for (int c = 0; c < comp; c++)
    // {
    // for (int i = threadIdx.x; i < input_tile_width; i++)
    // {
    //     for (int j = threadIdx.y; j < input_tile_width; j++)
    //     {
    //         if ((in_column + i) >= 0 && (in_column + i) < width && (in_row + j) >= 0 && (in_row + j) < height)
    //         {
    //             for (int c = 0; c < comp; c++)
    //             {
    //                 shared_input[(j * input_tile_width + i) * comp + c] = input_image[((in_row + j) * width + (in_column + i)) * comp + c];
    //             }
    //         }
    //         else
    //         {
    //             for (int c = 0; c < comp; c++)
    //             {

    //                 shared_input[(j * input_tile_width + i) * comp + c] = 0.0f;
    //             }
    //         }
    //     }
    // }
    // }
    int stride = OUTPUT_TILE_WIDTH * OUTPUT_TILE_WIDTH;
    for (int i = 0; i < elements_per_thread; i++) // for each thread iterate ove the elements that it should load => same element in eaxg row in the inpyt tile
    {
        // printf("i = %d\n", i);
        // the thread index with respect to the rest of the threads
        // if the output tile is 3x3 => 9 threads
        // the thread_index should go from 0-8
        int thread_index = threadIdx.y * OUTPUT_TILE_WIDTH + threadIdx.x;

        // printf("thread_index = %d\n", thread_index);

        // since a single thread loads more than one input element
        // the step is the difference between elements for which a single thread loads its in
        int thread_index_step = thread_index + (i * stride);

        // printf("i*stride = %d\n", i * stride);
        // printf("thread_index_step = %d\n", thread_index_step);

        // get the indices of the the thread with respect to the input tile
        // if a thread is number 8 and the input tile is 5x5 => then the thread loads into the cell 1,3 in the shared memory
        int shm_index_row = thread_index_step / input_tile_width;
        int shm_index_col = thread_index_step - (shm_index_row * input_tile_width);

        // printf("shm_index_row = %d, shm_index_col = %d\n", shm_index_row, shm_index_col);

        // printf("thread_index = %d, thread_index_step = %d, shm_index_row = %d, shm_index_col = %d\n", thread_index, thread_index_step, shm_index_row, shm_index_col);

        // get the index of the thread with respect to the input image
        // use the in_column and in_row => wrong
        // use the shared memory indices

        if (shm_index_col >= 0 && shm_index_col < input_tile_width && shm_index_row >= 0 && shm_index_row < input_tile_width)
        {
            int input_index_col = start_block_column + shm_index_col;
            int input_index_row = start_block_row + shm_index_row;

            if (input_index_col >= 0 && input_index_col < width && input_index_row >= 0 && input_index_row < height)
            {
                // printf("in_column + shm_index_col = %d, in_row + shm_index_row = %d\n", in_column + shm_index_col, in_row + shm_index_row);
                // printf("shm_index_col = %d, shm_index_row = %d\n", shm_index_col, shm_index_row);
                for (int c = 0; c < comp; c++)
                {
                    // printf("c = %d\n", (shm_index_row * input_tile_width + shm_index_col) * comp + c);
                    // printf("shm[%d] = %f\n", (shm_index_row * input_tile_width + shm_index_col) * comp + c, input_image[((in_row + shm_index_row) * width + (in_column + shm_index_col)) * comp + c]);
                    if (blockIdx.x == 0 && blockIdx.y == 0)
                    {
                        // printf("in_column = %d, in_row = %d\n", in_column, in_row );
                        printf("NORMAL CELL: in_column = %d, in_row = %d, shm_col = %d, shm_row = %d ,shm[%d] = %f, thread_index = %d\n", in_column, in_row, shm_index_col, shm_index_row, (shm_index_row * input_tile_width + shm_index_col) * comp + c, input_image[((start_block_row + shm_index_row) * width + (start_block_column + shm_index_col)) * comp + c], thread_index);
                    }
                    shared_input[(shm_index_row * input_tile_width + shm_index_col) * comp + c] = input_image[(input_index_row * width + input_index_col) * comp + c];
                }
            }
            else
            {
                for (int c = 0; c < comp; c++)
                {
                    // printf("c = %d\n", (shm_index_row * input_tile_width + shm_index_col) * comp + c);
                    if (blockIdx.x == 0 && blockIdx.y == 0)
                    {
                        printf("GHOST CELL: in_column = %d, in_row = %d ,shm[%d] = %f\n", in_column, in_row, (shm_index_row * input_tile_width + shm_index_col) * comp + c, 0.0f);
                    }

                    shared_input[(shm_index_row * input_tile_width + shm_index_col) * comp + c] = 0.0f;
                }
            }
        }
    }

    __syncthreads();

    // 2. Compute the output tile

    // all threads will participate in computing the pixel value
    // no need to check if the thread is within the boundaries
    if (out_column < width && out_row < height)
    {
        float pixel_value = 0.0f;
        for (int c = 0; c < comp; c++)
        {
            // iterate over the mask elements => surrounding box
            for (int mask_row = 0; mask_row < mask_size; mask_row++) // rows
            {
                for (int mask_column = 0; mask_column < mask_size; mask_column++) // columns
                {
                    // int in_row_index = in_row + mask_row;
                    // int in_col_index = in_column + mask_column;
                    // if (in_row_index >= 0 && in_row_index < height && in_col_index >= 0 && in_col_index < width)
                    // {
                    pixel_value += shared_input[((threadIdx.y + mask_row) * input_tile_width + (threadIdx.x + mask_column)) * comp + c] * constant_mask[mask_row * mask_size + mask_column];
                    // }
                }
            }
        }

        pixel_value = fminf(fmaxf(pixel_value, 0.0f), 1.0f); // clamp the pixel value to be in the range [0, 1]

        pixel_value = pixel_value * 255; // scale the pixel value to be in the range [0, 255]

        // 3. Write the output tile to the output image
        output_image[out_row * width + out_column] = (unsigned char)pixel_value;

        // TEST
        // load the data from the shared memory as is without any convolution
        // apply average to get the grayscale
        // float pixel_value = 0.0f;
        // for (int c = 0; c < comp; c++)
        // {
        //     pixel_value += shared_input[((threadIdx.y + 1) * input_tile_width + (threadIdx.x + 1)) * comp + c];
        // }

        // pixel_value = pixel_value / comp; // average

        // pixel_value = fminf(fmaxf(pixel_value, 0.0f), 1.0f); // clamp the pixel value to be in the range [0, 1]

        // pixel_value = pixel_value * 255; // scale the pixel value to be in the range [0, 255]

        // output_image[out_row * width + out_column] = (unsigned char)pixel_value;
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
    printf("mask_file_path = %s\n", mask_file_path);

    // 1. Allocate host memory => read input image / convolution mask

    // 1.1 Read image
    int width, height, comp;

    // get the fulle input path of the image
    char full_input_path[256];
    sprintf(full_input_path, "%s/%s", input_folder_path, "/tree.jpg");

    unsigned char *input_image = readImage(full_input_path, &width, &height, &comp);

    if (!input_image)
    {
        printf("Error: failed to read image\n");
        exit(1);
    }
    printf("width = %d, height = %d, comp = %d (channels)\n", width, height, comp); // print image dimensions
    int N = width * height * comp;

    printf("INPUT IMAGE\n");
    for (size_t i = 0; i < NUM_PIXELS_TO_PRINT * comp; i++)
    {
        printf("%d%s", input_image[i], ((i + 1) % comp) ? " " : "\n");
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

    // Allocate memory for the output image
    unsigned char *output_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    // 1.2 Read convolution masks
    FILE *mask_file = fopen(mask_file_path, "r");
    if (!mask_file)
    {
        printf("Error: failed to open mask file\n");
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

    const int input_tile_width = OUTPUT_TILE_WIDTH + mask_size - 1;
    printf("INPUT TILE WIDTH = %d\n", input_tile_width);
    printf("OUTPUT TILE WIDTH = %d\n", OUTPUT_TILE_WIDTH);

    // 2. Allocate device memory
    float *d_image;
    unsigned char *d_out;
    cudaMalloc((void **)&d_image, sizeof(float) * N);
    cudaMalloc((void **)&d_out, sizeof(unsigned char) * width * height);

    // 3. Transfer data from host to device memory
    cudaMemcpy(d_image, normalized_image, sizeof(float) * N, cudaMemcpyHostToDevice);

    // 4. Launch kernel
    dim3 block_dim(OUTPUT_TILE_WIDTH, OUTPUT_TILE_WIDTH);
    int grid_columns = ceil((float)width / OUTPUT_TILE_WIDTH);
    int grid_rows = ceil((float)height / OUTPUT_TILE_WIDTH);
    dim3 grid_dim(grid_columns, grid_rows);

    int shared_memory_size = input_tile_width * input_tile_width * comp * sizeof(float);
    printf("shared_memory_size = %d\n", shared_memory_size);

    kernel3<<<grid_dim, block_dim, shared_memory_size>>>(d_out, d_image, width, height, comp, mask_size, input_tile_width);

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
    int count = 0;
    for (size_t i = 0; i < NUM_PIXELS_TO_PRINT; i++)
    {
        printf("%d%s", output_image[i], "\n");
        count++;
    }
    printf("count = %d\n", count);

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