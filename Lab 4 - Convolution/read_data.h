#define STB_IMAGE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_NO_FAILURE_STRINGS

#include <stdio.h>
#include "./include/stb/stb_image.h"
#include "./include/dirent.h"

#define NUM_PIXELS_TO_PRINT 10
#define MAX_MASK_SIZE 10

// read rgb image given the file path
unsigned char *readImage(char const *filepath, int *width, int *height, int *comp)
{
    unsigned char *data = stbi_load(filepath, width, height, comp, 0);
    if (*data)
    {
        return data;
    }
    return NULL;
}

bool getImageDimensions(char *input_folder_path, int *width, int *height, int *channels)
{
    DIR *input_directory;
    struct dirent *entry;

    if ((input_directory = opendir(input_folder_path)) != NULL) // loop to make sure that the file read is an image
    {
        while ((entry = readdir(input_directory)) != NULL)
        {
            if (entry->d_type == DT_REG)
            {
                const char *image_name = entry->d_name;
                // Get full input path
                char full_input_path[256];
                sprintf(full_input_path, "%s/%s", input_folder_path, image_name);
                printf("FULL INPUT PATH: %s\n", full_input_path);

                // Read the image
                unsigned char *input_image = readImage(full_input_path, width, height, channels);

                if (input_image == NULL)
                {
                    printf("Error: failed to read image\n");
                    exit(1);
                }

                printf("width = %d, height = %d, channels = %d\n", *width, *height, *channels);

                break;
            }
        }

        closedir(input_directory);
        return true;
    }
    return false;
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
