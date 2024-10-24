#ifndef FFT2D_CUDA_H
#define FFT2D_CUDA_H

#include <fftw3.h>
#include <cufft.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Define complex data type
typedef float2 Complex;

// Helpers
__global__ void bitReversalPermutationKernel(Complex* data, int n, int log2n);
__global__ void transposeKernel(Complex* d_out, Complex* d_in, int width, int height);
void transpose_CUDA(Complex* d_out, Complex* d_in, int width, int height);
void normalize(Complex* data, int width, int height);

// FFT Functions
void fft1D(Complex* d_data, int n, int direction);
void fft2D(Complex* d_data, int width, int height, int direction);
void fft2D_FFTW(Complex* h_data, int width, int height, int direction);

// Reconstruction and Frame Saving
void reconstructAndSaveFrames(
    std::vector<Complex>& data_R,
    std::vector<Complex>& data_G,
    std::vector<Complex>& data_B,
    int width,
    int height,
    // const std::vector<int>& sorted_indices,
    const std::string& output_dir
);

#endif // FFT2D_CUDA_H
