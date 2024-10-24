#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fftw3.h>
#include <cufft.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono> // For timing
#include <fstream> // For report generation
#include <opencv2/opencv.hpp>
#include "fft2d_cuda.h"

// Find the symmetric index in the frequency domain
int findSymmetricIndex(int idx, int width, int height) {
    return (width * height) - idx - 1;
}

// Pixel brightness scaling
float get_rec_scale_factor(int idx, int totalComponents) {
    float c = log2f(3) / 2;
    float f = 1.0f / log2f((static_cast<float>(idx) / static_cast<float>(totalComponents)) + 2);
    return c * f;
}

// Function to save a frame using OpenCV
void saveFrame(const std::string& filename, cv::Mat& image) {
    if (!cv::imwrite(filename, image)) {
        std::cerr << "Error: Could not save image " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to normalize inverse FFT
void normalize(Complex* data, int width, int height) {
    for(int i = 0; i < width * height; i++) {
        data[i].x /= (width * height);
        data[i].y /= (width * height);
    }
}

// Function to clamp and convert float to uchar
uchar clampToUchar(float value, float scale_factor) {
    value = value * 255.0f * scale_factor; // Scale to 0-255
    if (value < 0.0f) return 0;
    if (value > 255.0f) return 255;
    return static_cast<uchar>(value);
}

// Error checking macro
#define CUDA_CHECK(error) \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Bit-reversal permutation kernel
__global__ void bitReversalPermutationKernel(Complex* data, int n, int log2n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;

    // Compute bit-reversed index
    unsigned int reversed = 0;
    unsigned int x = idx;
    for (int i = 0; i < log2n; i++) {
        reversed = (reversed << 1) | (x & 1);
        x >>= 1;
    }

    if (reversed > idx) {
        // Swap the elements
        Complex temp = data[idx];
        data[idx] = data[reversed];
        data[reversed] = temp;
    }
}

// 1D FFT Butterfly Kernel
__global__ void fft1D_kernel(Complex* data, int m, int direction, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int butterflies_per_group = m / 2;
    int total_groups = n / m;

    int total_butterflies = butterflies_per_group * total_groups;

    if (idx >= total_butterflies) return;

    int group = idx / butterflies_per_group;
    int butterfly = idx % butterflies_per_group;

    int even_index = group * m + butterfly;
    int odd_index = even_index + butterflies_per_group;

    // Bounds checking
    if (even_index >= n || odd_index >= n) return;

    Complex even = data[even_index];
    Complex odd = data[odd_index];

    float angle = -direction * 2.0f * M_PI * butterfly / m;
    float cos_angle = cosf(angle);
    float sin_angle = sinf(angle);

    Complex twiddle;
    twiddle.x = cos_angle * odd.x - sin_angle * odd.y;
    twiddle.y = sin_angle * odd.x + cos_angle * odd.y;

    // Butterfly operations
    data[even_index].x = even.x + twiddle.x;
    data[even_index].y = even.y + twiddle.y;
    data[odd_index].x = even.x - twiddle.x;
    data[odd_index].y = even.y - twiddle.y;
}

// Function to perform 1D FFT on device data
void fft1D(Complex* d_data, int n, int direction) {
    if (n <= 0 || (n & (n - 1)) != 0) { // Check if n is a power of 2
        std::cerr << "Error: FFT size must be a power of 2. Provided size: " << n << std::endl;
        exit(EXIT_FAILURE);
    }

    int log2n = static_cast<int>(log2f(static_cast<float>(n)));

    // Perform bit-reversal permutation on the device
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bitReversalPermutationKernel<<<blocks, threads>>>(d_data, n, log2n);
    cudaError_t err = cudaDeviceSynchronize();
    CUDA_CHECK(err);

    // Perform FFT iterations
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;          // Current FFT size (m = 2^s)
        int butterflies_per_group = m / 2;
        int num_groups = n / m;  // Number of FFT groups

        int total_butterflies = butterflies_per_group * num_groups;

        int threads_per_block = 256;
        int blocks_per_grid = (total_butterflies + threads_per_block - 1) / threads_per_block;

        // Launch the corrected butterfly kernel
        fft1D_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, m, direction, n);
        err = cudaDeviceSynchronize();
        CUDA_CHECK(err);
    }
}

// Transpose Kernel (Assuming square matrices for simplicity)
__global__ void transposeKernel(Complex* d_out, Complex* d_in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row

    if (x < width && y < height) {
        int idx_in = y * width + x;
        int idx_out = x * height + y;
        d_out[idx_out] = d_in[idx_in];
    }
}

// Function to perform matrix transpose on device
void transpose_CUDA(Complex* d_out, Complex* d_in, int width, int height) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, width, height);
    cudaError_t err = cudaDeviceSynchronize();
    CUDA_CHECK(err);
}

// Self-contained 2D FFT Function
void fft2D(Complex* h_data, int width, int height, int direction) {
    // Validate input dimensions
    if (width <= 0 || height <= 0 ||
        (width & (width - 1)) != 0 || (height & (height - 1)) != 0) {
        std::cerr << "Error: FFT dimensions must be powers of 2. Provided size: "
                  << width << "x" << height << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t size = width * height * sizeof(Complex);

    // Allocate device memory for input/output data
    Complex* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Perform FFT on rows
    for (int row = 0; row < height; row++) {
        Complex* d_row = d_data + row * width;
        fft1D(d_row, width, direction);
    }

    // Allocate device memory for transposed data
    Complex* d_transposed;
    CUDA_CHECK(cudaMalloc(&d_transposed, size));

    // Transpose the matrix
    transpose_CUDA(d_transposed, d_data, width, height);

    // Perform FFT on columns (now rows after transpose)
    for (int col = 0; col < width; col++) {
        Complex* d_col = d_transposed + col * height;
        fft1D(d_col, height, direction);
    }

    // Transpose back to original layout
    transpose_CUDA(d_data, d_transposed, height, width);

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_transposed));
}

// Function to perform FFTW-based 2D FFT
void fft2D_FFTW(Complex* h_data, int width, int height, int direction) {
    // Allocate FFTW input/output array
    fftwf_complex* fftw_data = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * width * height);
    if (!fftw_data) {
        std::cerr << "Error: FFTW memory allocation failed." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Copy h_data to fftw_data
    for (int i = 0; i < width * height; ++i) {
        fftw_data[i][0] = h_data[i].x; // Real part
        fftw_data[i][1] = h_data[i].y; // Imaginary part
    }

    // Create FFTW plan
    fftwf_plan plan;
    if (direction == 1) { // Forward FFT
        plan = fftwf_plan_dft_2d(height, width, fftw_data, fftw_data, FFTW_FORWARD, FFTW_ESTIMATE);
    } else { // Inverse FFT
        plan = fftwf_plan_dft_2d(height, width, fftw_data, fftw_data, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    if (!plan) {
        std::cerr << "Error: FFTW plan creation failed." << std::endl;
        fftwf_free(fftw_data);
        exit(EXIT_FAILURE);
    }

    // Execute FFTW
    fftwf_execute(plan);

    // Copy fftw_data back to h_data
    for (int i = 0; i < width * height; ++i) {
        h_data[i].x = fftw_data[i][0];
        h_data[i].y = fftw_data[i][1];
    }

    // Destroy FFTW plan and free memory
    fftwf_destroy_plan(plan);
    fftwf_free(fftw_data);
}

// Function to reconstruct and save frames
void reconstructAndSaveFrames(
    std::vector<Complex>& data_R,
    std::vector<Complex>& data_G,
    std::vector<Complex>& data_B,
    int width,
    int height,
    const std::string& output_dir
) {
    int totalComponents = width * height;

    // Initialize cumulative reconstructed image channels
    std::vector<float> rec_image_R(totalComponents, 0.0f);
    std::vector<float> rec_image_G(totalComponents, 0.0f);
    std::vector<float> rec_image_B(totalComponents, 0.0f);

    // Prepare frequency components and indices for each channel
    std::vector<std::pair<float, int>> freq_indices_R(totalComponents);
    std::vector<std::pair<float, int>> freq_indices_G(totalComponents);
    std::vector<std::pair<float, int>> freq_indices_B(totalComponents);

    // Populate frequency magnitudes and indices for each channel
    for (int i = 0; i < totalComponents; i++) {
        float freq_magnitude_R = std::sqrt(data_R[i].x * data_R[i].x + data_R[i].y * data_R[i].y);
        float freq_magnitude_G = std::sqrt(data_G[i].x * data_G[i].x + data_G[i].y * data_G[i].y);
        float freq_magnitude_B = std::sqrt(data_B[i].x * data_B[i].x + data_B[i].y * data_B[i].y);
        freq_indices_R[i] = {freq_magnitude_R, i};
        freq_indices_G[i] = {freq_magnitude_G, i};
        freq_indices_B[i] = {freq_magnitude_B, i};
    }

    // Sort frequency components by magnitude in descending order
    auto compare = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first > b.first;
    };
    std::sort(freq_indices_R.begin(), freq_indices_R.end(), compare);
    std::sort(freq_indices_G.begin(), freq_indices_G.end(), compare);
    std::sort(freq_indices_B.begin(), freq_indices_B.end(), compare);

    // OpenCV Mat for cumulative reconstructed image
    cv::Mat reconstructed(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    int channel = -1;
    int seconds = 6;
    int display_all_until = static_cast<int>(10 * (seconds * (1.0f / 3.0f)));
    int display_step = static_cast<int>(totalComponents / (10 * (seconds * (2.0f / 3.0f))));
    int next_display = display_all_until + display_step;

    // Initialize individual grating arrays for each channel
    std::vector<Complex> individual_grating_R(totalComponents, {0.0f, 0.0f});
    std::vector<Complex> individual_grating_G(totalComponents, {0.0f, 0.0f});
    std::vector<Complex> individual_grating_B(totalComponents, {0.0f, 0.0f});

    // Loop over frequency components
    for (int idx = 0; idx < totalComponents; idx++) {
        // For each channel, process the ith ranked frequency component
        // Process Red channel
        int idx_R = freq_indices_R[idx].second;
        int symm_idx_R = findSymmetricIndex(idx_R, width, height);

        individual_grating_R[idx_R] = data_R[idx_R];
        individual_grating_R[symm_idx_R] = data_R[symm_idx_R];

        // Process Green channel
        int idx_G = freq_indices_G[idx].second;
        int symm_idx_G = findSymmetricIndex(idx_G, width, height);

        individual_grating_G[idx_G] = data_G[idx_G];
        individual_grating_G[symm_idx_G] = data_G[symm_idx_G];

        // Process Blue channel
        int idx_B = freq_indices_B[idx].second;
        int symm_idx_B = findSymmetricIndex(idx_B, width, height);

        individual_grating_B[idx_B] = data_B[idx_B];
        individual_grating_B[symm_idx_B] = data_B[symm_idx_B];

        // Perform inverse FFT on individual gratings to get sine waves
        std::vector<float> rec_grating_R(totalComponents, 0.0f);
        std::vector<float> rec_grating_G(totalComponents, 0.0f);
        std::vector<float> rec_grating_B(totalComponents, 0.0f);

        fft2D(individual_grating_R.data(), width, height, -1);
        fft2D(individual_grating_G.data(), width, height, -1);
        fft2D(individual_grating_B.data(), width, height, -1);
        normalize(individual_grating_R.data(), width, height);
        normalize(individual_grating_G.data(), width, height);
        normalize(individual_grating_B.data(), width, height);

        // Add sine waves to cumulative reconstruction
        float max_brightness_pixels[3] = {0.0f, 0.0f, 0.0f};
        for (int i = 0; i < totalComponents; i++) {
            rec_image_R[i] += individual_grating_R[i].x;
            rec_image_G[i] += individual_grating_G[i].x;
            rec_image_B[i] += individual_grating_B[i].x;
            max_brightness_pixels[0] = std::max(max_brightness_pixels[0], individual_grating_R[i].x);
            max_brightness_pixels[1] = std::max(max_brightness_pixels[1], individual_grating_G[i].x);
            max_brightness_pixels[2] = std::max(max_brightness_pixels[2], individual_grating_B[i].x);
        }

        // Combine channels into the reconstructed image
        float rec_scale_factor = get_rec_scale_factor(idx, totalComponents); // Darken
        for (int i = 0; i < totalComponents; i++) {
            int y = i / width;
            int x = i % width;
            uchar r = clampToUchar(rec_image_R[i], rec_scale_factor);
            uchar g = clampToUchar(rec_image_G[i], rec_scale_factor);
            uchar b = clampToUchar(rec_image_B[i], rec_scale_factor);
            reconstructed.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r); // OpenCV uses BGR
        }

        // Save frames based on criteria
        if (idx < display_all_until || idx == next_display) {
            channel = (channel + 1) % 3;
            std::cout << "Index: " << idx << "/" << totalComponents << std::endl;
            if (idx >= display_all_until && idx == next_display) {
                next_display += display_step;
            }

            // Save the cumulative reconstruction image
            std::string cumulative_filename = output_dir + "/step_" + std::to_string(idx) + "_cumulative.png";
            saveFrame(cumulative_filename, reconstructed);

            // Save the individual grating image
            cv::Mat grating_image(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
            float individual_scale_factor = 255.0f / max_brightness_pixels[channel]; // Brighten
            for (int i = 0; i < totalComponents; i++) {
                int y = i / width;
                int x = i % width;
                uchar val_R = (channel == 0) ? clampToUchar(individual_grating_R[i].x, individual_scale_factor) : 0;
                uchar val_G = (channel == 1) ? clampToUchar(individual_grating_G[i].x, individual_scale_factor) : 0;
                uchar val_B = (channel == 2) ? clampToUchar(individual_grating_B[i].x, individual_scale_factor) : 0;
                grating_image.at<cv::Vec3b>(y, x) = cv::Vec3b(val_B, val_G, val_R);
            }
            std::string grating_filename = output_dir + "/step_" + std::to_string(idx) + "_channel.png";
            saveFrame(grating_filename, grating_image);
        }

        // Reset individual grating arrays for next iteration
        individual_grating_R[idx_R] = {0.0f, 0.0f};
        individual_grating_R[symm_idx_R] = {0.0f, 0.0f};

        individual_grating_G[idx_G] = {0.0f, 0.0f};
        individual_grating_G[symm_idx_G] = {0.0f, 0.0f};

        individual_grating_B[idx_B] = {0.0f, 0.0f};
        individual_grating_B[symm_idx_B] = {0.0f, 0.0f};
    }

    // Save the final reconstructed image
    for (int i = 0; i < totalComponents; i++) {
        int y = i / width;
        int x = i % width;

        uchar r = clampToUchar(rec_image_R[i], 0.5f);
        uchar g = clampToUchar(rec_image_G[i], 0.5f);
        uchar b = clampToUchar(rec_image_B[i], 0.5f);

        reconstructed.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
    }
    std::string final_filename = output_dir + "/reconstructed_final.png";
    saveFrame(final_filename, reconstructed);
}

