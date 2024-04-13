#define STB_IMAGE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_NO_FAILURE_STRINGS

#include <stdio.h>
#include "./include/stb/stb_image.h"

#define NUM_PIXELS_TO_PRINT 10
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 256
#define CHANNELS 3
#define MAX_MASK_SIZE 10

// read rgb image given the file path
unsigned char *readImage(char const *filepath, int *width, int *height, int *comp)
{
    unsigned char *data = stbi_load(filepath, width, height, comp, 0);
    if (*data)
    {
        // for (size_t i = 0; i < NUM_PIXELS_TO_PRINT * (*comp); i++)
        // {
        //     printf("%d%s", data[i], ((i + 1) % (*comp)) ? " " : "\n");
        // }
        // printf("\n");
        return data;
    }
    return NULL;
}

// read mask elements from file
int readMaskSize(FILE *input)
{
    int mask_size;
    fscanf(input, "%d", &mask_size);
    printf("Mask Size = %d\n", mask_size);
    char endline = getc(input);
    return mask_size;
}

// read a single mask for a single channel
void readMask(FILE *input, float *mask, int mask_size)
{
    char endline;

    // read matrix elements
    for (int i = 0; i < mask_size * mask_size; i++)
    {
        if ((i + 1) % mask_size == 0)
        {
            fscanf(input, "%f", &mask[i]);

            endline = getc(input);
            if (endline == '\n')
            {
                continue;
            }
        }
        fscanf(input, "%f", &mask[i]);
    }
}

// print a given mask
void printMask(float *mask, int mask_size)
{
    for (int i = 0; i < mask_size * mask_size; i++)
    {
        printf("%f ", mask[i]);
        if ((i + 1) % mask_size == 0)
        {
            printf("\n");
        }
    }
}

int readCommandLineArguments(int argc, char *argv[], char **input_image_path, char **output_image_path, char **mask_path)
{

    if (argc < 5)
    {
        printf("Usage: %s <input_image_path> <output_image_path> <mask_path>\n", argv[0]);
        exit(1);
    }

    *input_image_path = argv[1];
    *output_image_path = argv[2];
    int batch_size = atoi(argv[3]);
    *mask_path = argv[4];

    return batch_size;
}
