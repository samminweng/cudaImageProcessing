#include <stdio.h>
#include <tuple>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
const int THREADS_PER_BLOCK = 256;

typedef struct Pixel {
    stbi_uc r;
    stbi_uc g;
    stbi_uc b;
    stbi_uc a;
} Pixel;


void imageFree(stbi_uc* image);
void writeImage(const char* path_to_image, stbi_uc* image, int width, int height, int channels);
stbi_uc* loadImage(const char* path_to_image, int* width, int* height, int* channels);
__global__ void blurKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads, int padded_width, int padded_height, int* mask, int mask_size);
__host__ __device__ void getPixel(stbi_uc* image, int width, int x, int y, Pixel* pixel);
__host__ __device__ void setPixel(stbi_uc* image, int width, int x, int y, Pixel* pixel);
__device__ int clamp_int(double color);