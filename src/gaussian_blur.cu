#include "gaussian_blur.h"

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}
// Load the image
stbi_uc* loadImage(const char* path_to_image, int* width, int* height, int* channels) {
    return stbi_load(path_to_image, width, height, channels, STBI_rgb_alpha);
}
// Get the pixel at (x, y)
__host__ __device__ void getPixel(stbi_uc* image, int width, int x, int y, Pixel* pixel) {
    const stbi_uc* p = image + (STBI_rgb_alpha * (y * width + x));
    pixel->r = p[0];
    pixel->g = p[1];
    pixel->b = p[2];
    pixel->a = p[3];
}
// Set the pixel at (x, y)
__host__ __device__ void setPixel(stbi_uc* image, int width, int x, int y, Pixel* pixel) {
    stbi_uc* p = image + (STBI_rgb_alpha * (y * width + x));
    p[0] = pixel->r;
    p[1] = pixel->g;
    p[2] = pixel->b;
    p[3] = pixel->a;
}
// Print the pixel
__host__ __device__ void printPixel(Pixel* pixel) {
    printf("r = %u, g = %u, b = %u, a = %u\n", pixel->r, pixel->g, pixel->b, pixel->a);
}

// Free image 
void imageFree(stbi_uc* image) {
    stbi_image_free(image);
}

void writeImage(const char* path_to_image, stbi_uc* image, int width, int height, int channels) {
    stbi_write_png(path_to_image, width, height, channels, image, width * channels);
}


// Create a padded input image with three channels 
stbi_uc* zeroPadImage(stbi_uc* input_image, int &width, int &height, int channels, int filter_size) {
    int half_filter_size = filter_size / 2;
    int padded_width = width + 2 * half_filter_size;
    int padded_height = height + 2 * half_filter_size;

    stbi_uc* padded_image = (stbi_uc*) malloc(channels * padded_width * padded_height * sizeof(stbi_uc));

    Pixel zero_pixel = { .r = 0, .g = 0, .b = 0, .a = 0 };
    Pixel other_pixel;
    // Set up 
    for (int i = 0; i < padded_width; i++) {
        for (int j = 0; j < padded_height; j++) {
            if (i < half_filter_size || i > padded_width - half_filter_size || j < half_filter_size || j > padded_width - half_filter_size) {
                setPixel(padded_image, padded_width, i, j, &zero_pixel);
            } else {
                getPixel(input_image, width, i - half_filter_size, j - half_filter_size, &other_pixel);
                setPixel(padded_image, padded_width, i, j, &other_pixel);
            }
        }
    }

    width = padded_width;
    height = padded_height;

    return padded_image;
}


// Blurred Kernel
__global__ void blurKernel(stbi_uc* input_image, stbi_uc* output_image, int width, int height, int channels, int total_threads, int padded_width, int padded_height, int* mask, int mask_size) {
    const int thread_id = threadIdx.x + blockDim.x * blockIdx.x; // Get thread ID
    if (thread_id >= total_threads) {
        return;
    }

    int padding_size = mask_size / 2;

    int y_coordinate = (thread_id / width) + padding_size;
    int x_coordinate = (thread_id % height) + padding_size;

    Pixel current_pixel;
    double red = 0;
    double blue = 0;
    double green = 0;
    double alpha = 0;
    double blur_coef = 1.0f/256;
    // Apply mask filter on the curent pixel (r, g, b channels)
    for (int i = 0; i < mask_size; i++) {
        for (int j = 0; j < mask_size; j++) {
            // Get the current pixel from the padded image
            getPixel(input_image, padded_width, x_coordinate - padding_size + i, y_coordinate - padding_size + j, &current_pixel);
            int mask_element = mask[i * mask_size + j];
            red += current_pixel.r * mask_element * blur_coef;
            green += current_pixel.g * mask_element * blur_coef;
            blue += current_pixel.b * mask_element * blur_coef;
            alpha += current_pixel.a * mask_element * blur_coef;
        }
    }

    Pixel pixel;
    pixel.r = clamp_int(red);
    pixel.g = clamp_int(green);
    pixel.b = clamp_int(blue);
    pixel.a = clamp_int(alpha);
    // Set the current pixel with new pixel
    setPixel(output_image, width, x_coordinate, y_coordinate, &pixel);
    
}

// Get a range of color value from 0 to 255
__device__ int clamp_int(double color){
    if(color <0){
        return 0;
    }else if(color > 255){
        return 255;
    }
    return int(color);
}

// Perform the blurring image
stbi_uc* blurImage(stbi_uc* input_image, int width, int height, int channels) {
    // Create mask filters
    int MASK_SIZE = 5;
    int mask[MASK_SIZE*MASK_SIZE] = {1, 4, 6, 4, 1,
                  4, 16, 24, 16, 4, 
                  6, 24, 36, 24, 6,
                  4, 16, 24, 16, 4,
                  1, 4, 6, 4, 1
                };
    
    // for(int i = 0; i < MASK_SIZE*MASK_SIZE; i++) {
        // mask[i] = 1;
    // }
    // Move the mask to GPU device
    int* d_mask;
    cudaMallocManaged(&d_mask, MASK_SIZE * MASK_SIZE * sizeof(int));
    cudaMemcpy(d_mask, mask, MASK_SIZE * MASK_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    // Create an padded input iage with padding 
    int padded_width = width;
    int padded_height = height;
    stbi_uc* padded_image = zeroPadImage(input_image, padded_width, padded_height, channels, MASK_SIZE);
    // Create the output image (blurred image)
    int image_size = channels * padded_width * padded_height * sizeof(stbi_uc);
    
    
    // Create device input and output images
    stbi_uc* d_input_image;
    cudaMallocManaged(&d_input_image, image_size);
    stbi_uc* d_output_image;
    cudaMallocManaged(&d_output_image, image_size);
    // Copy host padded image to device input image
    cudaMemcpy(d_input_image, padded_image, image_size, cudaMemcpyHostToDevice);
    imageFree(padded_image); // Remove the host padded image

    int total_threads = width * height;
    int threads = min(THREADS_PER_BLOCK, total_threads); // Thread in each block
    // The total number of blocks
    int blocks = total_threads / THREADS_PER_BLOCK;
    printf("Total number of Blocks = %d, Number of threads per block = %d\n", blocks, threads);
    // Run the cuda blur Kernel with 
    blurKernel<<<blocks, threads>>>(d_input_image, d_output_image, width, height, channels, 
                                    total_threads, padded_width, padded_height, d_mask, MASK_SIZE);
    cudaDeviceSynchronize();

    //Copy the device output image to host
    stbi_uc* h_output_image = (stbi_uc*) malloc(image_size);
    for (int i = 0; i < padded_width * padded_height; i++) {
        h_output_image[i] = input_image[i];
    }
    cudaMemcpy(h_output_image, d_output_image, image_size, cudaMemcpyDeviceToHost);
    return h_output_image;

}

int main(int argc, const char* argv[]) {
    // cuda_hello<<<1,1>>>(); 
    // cudaDeviceSynchronize();
    // const char* path_to_input_image = "images/lena_rgb.png";
    // const char* path_to_output_image = "images/lena_rgb_blurred.png";
    if (argc != 6) {
        printf("Incorrect number of arguments.\n");
        return 1;
    }

    const char* path_to_input_image = argv[1];
    const char* path_to_output_image = argv[2];
    int width = atoi(argv[3]);
    int height = atoi(argv[4]);
    int channels = atoi(argv[5]);
    /*
	 * Load image into OpenCV matrix 
	*/
   
    stbi_uc* image = loadImage(path_to_input_image, &width, &height, &channels);
    if (image == NULL) {
        printf("Could not load image %s.\n", path_to_input_image);
        return 1;
    }
    printf("Load the images %s.\n", path_to_input_image);

    // Perform guassian blur
    stbi_uc* blurred_image;
    blurred_image = blurImage(image, width, height, channels);
    printf("Complete blurring operation on image\n");
    
    writeImage(path_to_output_image, blurred_image, width, height, channels);
    printf("Write the image to %s.\n", path_to_output_image);
    imageFree(image);
    imageFree(blurred_image);

    return 0;
}