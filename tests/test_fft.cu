#include <gtest/gtest.h>
#include "../fft2d_cuda.h"
#include "../image_processing.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fftw3.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

// Error checking macro
#define CUDA_CHECK(error) \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

//
void transpose(Complex* dst, const Complex* src, int width, int height) {
    for(int y = 0; y < height; ++y) {
        for(int x = 0; x < width; ++x) {
            dst[x * height + y] = src[y * width + x];
        }
    }
}

// Helper function to verify FFT results
bool verifyFFT(const std::vector<Complex>& computed, const std::vector<Complex>& expected, float epsilon = 1e-3f) {
    for(int i = 0; i < computed.size(); ++i) {
        if(std::abs(computed[i].x - expected[i].x) > epsilon ||
           std::abs(computed[i].y - expected[i].y) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": "
                      << "Computed(" << computed[i].x << ", " << computed[i].y << ") vs "
                      << "Expected(" << expected[i].x << ", " << expected[i].y << ")\n";
            return false;
        }
    }
    return true;
}

// 1 dimensional FFT with FFTW
void fft1D_FFTW(Complex* h_data, int n, int direction) {
    if (n <= 0 || (n & (n - 1)) != 0) { // Check if n is a power of 2
        std::cerr << "Error: FFT size must be a power of 2. Provided size: " << n << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate FFTW input/output array
    fftwf_complex* fftw_data = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n);
    if (!fftw_data) {
        std::cerr << "Error: FFTW memory allocation failed." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Copy h_data to fftw_data
    for (int i = 0; i < n; ++i) {
        fftw_data[i][0] = h_data[i].x; // Real part
        fftw_data[i][1] = h_data[i].y; // Imaginary part
    }
    
    // Create FFTW plan
    fftwf_plan plan;
    if (direction == 1) { // Forward FFT
        plan = fftwf_plan_dft_1d(n, fftw_data, fftw_data, FFTW_FORWARD, FFTW_ESTIMATE);
    } else { // Inverse FFT
        plan = fftwf_plan_dft_1d(n, fftw_data, fftw_data, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    
    if (!plan) {
        std::cerr << "Error: FFTW plan creation failed." << std::endl;
        fftwf_free(fftw_data);
        exit(EXIT_FAILURE);
    }
    
    // Execute FFTW
    fftwf_execute(plan);
    
    // Normalize if inverse FFT
    if (direction == -1) {
        for (int i = 0; i < n; ++i) {
            fftw_data[i][0] /= static_cast<float>(n);
            fftw_data[i][1] /= static_cast<float>(n);
        }
    }
    
    // Copy fftw_data back to h_data
    for (int i = 0; i < n; ++i) {
        h_data[i].x = fftw_data[i][0];
        h_data[i].y = fftw_data[i][1];
    }
    
    // Destroy FFTW plan and free memory
    fftwf_destroy_plan(plan);
    fftwf_free(fftw_data);
}

// 2 dimensional FFT that uses FFTW 1D implementation
void fft2D_CustomFFTW(Complex* h_data, int width, int height, int direction) {
    // Validate input dimensions
    if (width <= 0 || height <= 0 ||
        (width & (width - 1)) != 0 || (height & (height - 1)) != 0) {
        std::cerr << "Error: FFT dimensions must be powers of 2. Provided size: "
                  << width << "x" << height << std::endl;
        exit(EXIT_FAILURE);
    }

    // Temporary buffer for transposed data
    std::vector<Complex> transposed_data(width * height);
    
    // Perform FFT on rows
    for(int y = 0; y < height; ++y) {
        Complex* row = h_data + y * width;
        fft1D_FFTW(row, width, direction);
    }

    // Transpose the matrix to perform FFT on columns
    transpose(transposed_data.data(), h_data, width, height);

    // Perform FFT on "columns" (which are now rows after transpose)
    for(int x = 0; x < width; ++x) {
        Complex* column = transposed_data.data() + x * height;
        fft1D_FFTW(column, height, direction);
    }

    // Transpose back to original layout
    transpose(h_data, transposed_data.data(), height, width);
}

// Test 1: Verify that image loading and cropping works correctly
TEST(ImageProcessingTest, LoadAndCropImage) {
    cv::Mat image;
    int width, height;
    bool result = loadAndCropImage("../test_images/sample.jpg", image, width, height);
    ASSERT_TRUE(result);
    EXPECT_EQ(width, height); // Image should be square
    EXPECT_FALSE(image.empty());
}

// Test 2: Verify accuracy of transpose operation
TEST(TransposeTest, CorrectTransposition) {
    int width = 2;
    int height = 2;
    std::vector<Complex> host_in = { {1,0}, {2,0}, {3,0}, {4,0} }; // 2x2 matrix
    std::vector<Complex> expected_out = { {1,0}, {3,0}, {2,0}, {4,0} }; // Transposed

    // Allocate device memory and copy input
    Complex* d_in;
    Complex* d_out;
    cudaMalloc(&d_in, 4 * sizeof(Complex));
    cudaMalloc(&d_out, 4 * sizeof(Complex));
    cudaMemcpy(d_in, host_in.data(), 4 * sizeof(Complex), cudaMemcpyHostToDevice);

    // Launch transpose kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, width, height);
    cudaDeviceSynchronize();

    // Copy back and verify
    std::vector<Complex> host_out(4);
    cudaMemcpy(host_out.data(), d_out, 4 * sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    for(int i = 0; i < 4; ++i) {
        EXPECT_EQ(host_out[i].x, expected_out[i].x);
        EXPECT_EQ(host_out[i].y, expected_out[i].y);
    }
}

// Test 3: Verify accuracy of bit reversal operation
TEST(BitReversalTest, CorrectReordering) {
    // Initialize data with distinct values
    int n = 8;
    std::vector<Complex> host_data(n);
    for(int i = 0; i < n; ++i) {
        host_data[i].x = static_cast<float>(i);
        host_data[i].y = 0.0f;
    }

    // Allocate device memory and copy data
    Complex* d_data;
    cudaMalloc(&d_data, n * sizeof(Complex));
    cudaMemcpy(d_data, host_data.data(), n * sizeof(Complex), cudaMemcpyHostToDevice);

    // Launch bit-reversal kernel
    int log2n = 3;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bitReversalPermutationKernel<<<blocks, threads>>>(d_data, n, log2n);
    cudaDeviceSynchronize();

    // Copy back and verify
    std::vector<Complex> reordered(n);
    cudaMemcpy(reordered.data(), d_data, n * sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Expected bit-reversed order for n=8: [0,4,2,6,1,5,3,7]
    int expected_order[8] = {0,4,2,6,1,5,3,7};
    for(int i = 0; i < n; ++i) {
        EXPECT_EQ(reordered[i].x, expected_order[i]);
        EXPECT_EQ(reordered[i].y, 0.0f);
    }
}

// Test 4: Isolate 2D FFT-specific operations from 1D FFt and verify accuracy
TEST(FFTConsistencyTest, CustomFFTWInverseFFT) {
    // Define FFT dimensions (must be powers of 2)
    int width = 8;
    int height = 8;

    // Create a simple test image (e.g., a white square on black background)
    cv::Mat test_image = cv::Mat::zeros(height, width, CV_32F);
    cv::rectangle(test_image, cv::Point(2, 2), cv::Point(5, 5), cv::Scalar(1.0f), -1);

    // Initialize data as row-major order
    std::vector<Complex> original_data(width * height, make_float2(0.0f, 0.0f));
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int idx = y * width + x;
            original_data[idx].x = test_image.at<float>(y, x);
            original_data[idx].y = 0.0f;
        }
    }
    
    // Prepare data for FFT
    std::vector<Complex> data_Custom = original_data;

    // Perform forward 2D FFT using CustomFFTW
    fft2D_CustomFFTW(data_Custom.data(), width, height, 1); // 1 for forward FFT
    
    // Perform inverse 2D FFT using CustomFFTW
    fft2D_CustomFFTW(data_Custom.data(), width, height, -1); // -1 for inverse FFT

    // Compare the final data with the original data
    EXPECT_TRUE(verifyFFT(data_Custom, original_data)) << "CustomFFTW Inverse FFT does not match original data.";

    // Optionally, print success message
    std::cout << "CustomFFTW Inverse FFT matches original data within tolerance." << std::endl;
}

// Test 5: Verify accuracy of 1D FFT
TEST(FFTConsistencyTest, CUDAFFTInverseFFT1D) {
    // Define FFT size (power of 2)
    int n = 8;

    // Initialize test data with a known pattern (e.g., a sine wave)
    std::vector<Complex> original_data(n, {0.0f, 0.0f});
    for(int i = 0; i < n; ++i) {
        original_data[i].x = std::sin(2.0f * M_PI * i / n);
        original_data[i].y = 0.0f;
    }

    // Prepare data
    std::vector<Complex> data_CUDA = original_data;
    std::vector<Complex> data_FFTW = original_data;

    // Allocate and copy memory for CUDA FFT
    Complex* d_data;
    cudaError_t err = cudaMalloc(&d_data, n * sizeof(Complex));
    ASSERT_EQ(err, cudaSuccess) << "CUDA malloc failed.";
    err = cudaMemcpy(d_data, data_CUDA.data(), n * sizeof(Complex), cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "CUDA memcpy to device failed.";

    // Perform forward FFT using CUDA
    fft1D(d_data, n, 1); // 1 for forward FFT
    err = cudaMemcpy(data_CUDA.data(), d_data, n * sizeof(Complex), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "CUDA memcpy to host failed.";
    cudaFree(d_data);

    // Perform forward FFT using FFTW
    fft1D_FFTW(data_FFTW.data(), n, 1); // 1 for forward FFT

    // Compare CUDA FFT results with FFTW results
    EXPECT_TRUE(verifyFFT(data_CUDA, data_FFTW)) << "FFT (CUDA) does not match FFT (FFTW).";

    // Allocate and copy memory for CUDA inverse FFT
    err = cudaMalloc(&d_data, n * sizeof(Complex));
    ASSERT_EQ(err, cudaSuccess) << "CUDA malloc failed for inverse FFT.";
    err = cudaMemcpy(d_data, data_CUDA.data(), n * sizeof(Complex), cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "CUDA memcpy to device failed for inverse FFT.";

    // Perform inverse FFT using CUDA
    fft1D(d_data, n, -1); // -1 for inverse FFT
    err = cudaMemcpy(data_CUDA.data(), d_data, n * sizeof(Complex), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "CUDA memcpy to host failed for inverse FFT.";
    cudaFree(d_data);

    // Normalize CUDA inverse FFT
    for(int i = 0; i < n; i++) {
        data_CUDA[i].x /= static_cast<float>(n);
        data_CUDA[i].y /= static_cast<float>(n);
    }

    // Perform inverse FFT using FFTW
    fft1D_FFTW(data_FFTW.data(), n, -1); // -1 for inverse FFT

    // Compare CUDA inverse FFT results with original data
    EXPECT_TRUE(verifyFFT(data_CUDA, original_data)) << "Inverse FFT (CUDA) does not match original data.";
    EXPECT_TRUE(verifyFFT(data_FFTW, original_data)) << "Inverse FFT (FFTW) does not match original data.";
}

// Test 6: Verify accuracy of 2D FFt
TEST(FFTConsistencyTest, CUDAFFTInverseFFT2D) {
    // Similar to FFTW test but using CUDA FFT implementation
    // Create a simple test image (e.g., a white square on black background)
    int width = 8;
    int height = 8;
    cv::Mat test_image = cv::Mat::zeros(height, width, CV_32F);
    cv::rectangle(test_image, cv::Point(2, 2), cv::Point(5, 5), cv::Scalar(1.0f), -1);

    // Separate channels (single channel for simplicity)
    std::vector<Complex> original_data(width * height, {0.0f, 0.0f});
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int idx = y * width + x;
            original_data[idx].x = test_image.at<float>(y, x);
            original_data[idx].y = 0.0f;
        }
    }
    
    std::vector<Complex> data_CUDA = original_data; // Keep original data for comparison
    std::vector<Complex> data_FFTW = original_data;

    fft2D(data_CUDA.data(), width, height, 1); // Forward FFT with CUDA
    fft2D_FFTW(data_FFTW.data(), width, height, 1); // Forward FFT with FFTW

    EXPECT_TRUE(verifyFFT(data_CUDA, data_FFTW)) << "Forward FFT results do not match FFTW.";

    fft2D(data_CUDA.data(), width, height, -1); // Inverse FFT with CUDA
    normalize(data_CUDA.data(), width, height);
    fft2D_FFTW(data_FFTW.data(), width, height, -1); // Inverse FFT with FFTW
    normalize(data_FFTW.data(), width, height);
    
    // Verify inverse FFT consistency with original data
    EXPECT_TRUE(verifyFFT(data_CUDA, original_data)) << "Inverse FFT (CUDA) does not match original data.";
    EXPECT_TRUE(verifyFFT(data_FFTW, original_data)) << "Inverse FFT (FFTW) does not match original data.";

}
