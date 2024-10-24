#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdlib> // For system()
#include <filesystem> // C++17 for directory creation
#include <chrono>
#include <fstream>
#include "image_processing.h"
#include "fft2d_cuda.h"

namespace fs = std::filesystem;

// Function to upload animation and report to S3 using AWS CLI
void uploadToS3(const std::string& file_path, const std::string& bucket, const std::string& key) {
    std::string command = "aws s3 cp " + file_path + " s3://" + bucket + "/" + key;
    int ret = system(command.c_str());
    if (ret != 0) {
        std::cerr << "Error: Failed to upload " << file_path << " to S3." << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to create output directory if it doesn't exist
bool createOutputDirectory(const std::string& dir_path) {
    try {
        if (!fs::exists(dir_path)) {
            fs::create_directories(dir_path);
        }
        return true;
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error: Could not create directory " << dir_path << ". " << e.what() << std::endl;
        return false;
    }
}

template <typename Func>
double measureTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func(); // Execute the function
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count(); // Return elapsed time in milliseconds
}

void generateReport(const std::string& reportFile, const double cudaTimes[3], const double fftwTimes[3]) {
    // Calculate speedup factors
    double speedups[3];
    for (int i = 0; i < 3; ++i) {
        speedups[i] = cudaTimes[i] > 0.0 ? fftwTimes[i] / cudaTimes[i] : 0.0;
    }

    // Calculate totals
    double totalCuda = cudaTimes[0] + cudaTimes[1] + cudaTimes[2];
    double totalFftw = fftwTimes[0] + fftwTimes[1] + fftwTimes[2];
    double totalSpeedup = cudaTimes[0] + cudaTimes[1] + cudaTimes[2] > 0.0 ? totalFftw / totalCuda : 0.0;

    // Open the report file
    std::ofstream ofs(reportFile);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open " << reportFile << " for writing." << std::endl;
        return;
    }

    // Write the table header
    ofs << "-------------------------------------------------------\n";
    ofs << "|  Channel  |  CUDA (ms)  |  FFTW (ms)  |  Ratio      |\n";
    ofs << "-------------------------------------------------------\n";

    // Define channel names
    const char* channels[3] = { "    RED", "  GREEN", "   BLUE" };

    // Write data for each channel
    for (int i = 0; i < 3; ++i) {
        ofs << "|  " << channels[i] << "  |  "
            << std::fixed << std::setprecision(7) << cudaTimes[i] << "  |  "
            << fftwTimes[i] << "  |  "
            << speedups[i] << "  |\n";
    }

    // Write totals
    ofs << "-------------------------------------------------------\n";
    ofs << "|    TOTAL  |  "
        << std::fixed << std::setprecision(7) << totalCuda << "  |  "
        << totalFftw << "  |  "
        << totalSpeedup << "  |\n";
    ofs << "-------------------------------------------------------\n";

    // Close the file
    ofs.close();

    std::cout << "Performance report generated: " << reportFile << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./fft_processor <input_image> <output_directory>" << std::endl;
        return -1;
    }

    std::string input_image = argv[1];
    std::string output_dir = argv[2];

    // Create output directory
    std::time_t t = std::time(nullptr);
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", std::localtime(&t));
    std::string timestamp_str = std::string(timestamp);
    
    if (!createOutputDirectory(output_dir)) {
        return -1;
    }

    // Load and crop the image
    cv::Mat image;
    int width, height;
    if (!loadAndCropImage(input_image, image, width, height)) {
        return -1;
    }

    // Convert image to float
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    
    int total = width * height;

    std::vector<Complex> data_R_CUDA(total, {0.0f, 0.0f});
    std::vector<Complex> data_G_CUDA(total, {0.0f, 0.0f});
    std::vector<Complex> data_B_CUDA(total, {0.0f, 0.0f});

    std::vector<Complex> data_R_FFTW(total, {0.0f, 0.0f});
    std::vector<Complex> data_G_FFTW(total, {0.0f, 0.0f});
    std::vector<Complex> data_B_FFTW(total, {0.0f, 0.0f});

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            cv::Vec3f pixel = image.at<cv::Vec3f>(y, x); // BGR
            data_R_CUDA[idx].x = pixel[2]; // Red
            data_R_CUDA[idx].y = 0.0f;
            data_G_CUDA[idx].x = pixel[1]; // Green
            data_G_CUDA[idx].y = 0.0f;
            data_B_CUDA[idx].x = pixel[0]; // Blue
            data_B_CUDA[idx].y = 0.0f;

            // Initialize FFTW data
            data_R_FFTW[idx].x = pixel[2];
            data_R_FFTW[idx].y = 0.0f;
            data_G_FFTW[idx].x = pixel[1];
            data_G_FFTW[idx].y = 0.0f;
            data_B_FFTW[idx].x = pixel[0];
            data_B_FFTW[idx].y = 0.0f;
        }
    }

    // Arrays to store elapsed times
    double cudaTimes[3] = { 0.0, 0.0, 0.0 };   // RED, GREEN, BLUE
    double fftwTimes[3] = { 0.0, 0.0, 0.0 };   // RED, GREEN, BLUE

    // Perform FFTs using CUDA
    std::cout << "Performing CUDA FFTs..." << std::endl;
    cudaTimes[0] = measureTime([&]() { fft2D(data_R_CUDA.data(), width, height, 1); }); // Red
    cudaTimes[1] = measureTime([&]() { fft2D(data_G_CUDA.data(), width, height, 1); }); // Green
    cudaTimes[2] = measureTime([&]() { fft2D(data_B_CUDA.data(), width, height, 1); }); // Blue
    std::cout << "CUDA FFTs completed." << std::endl;

    // Perform FFTs using FFTW
    std::cout << "Performing FFTW FFTs..." << std::endl;
    fftwTimes[0] = measureTime([&]() { fft2D_FFTW(data_R_FFTW.data(), width, height, 1); }); // Red
    fftwTimes[1] = measureTime([&]() { fft2D_FFTW(data_G_FFTW.data(), width, height, 1); }); // Green
    fftwTimes[2] = measureTime([&]() { fft2D_FFTW(data_B_FFTW.data(), width, height, 1); }); // Blue
    std::cout << "FFTW FFTs completed." << std::endl;

    // Generate performance report
    generateReport("performance_report.txt", cudaTimes, fftwTimes);

    // Reconstruct and save frames
    std::cout << "Reconstructing image and saving frames..." << std::endl;
    reconstructAndSaveFrames(data_R_CUDA, data_G_CUDA, data_B_CUDA, width, height, output_dir);

    // Create animation
    std::cout << "Creating animation from frames..." << std::endl;
    std::string python_command = "python3 create_animation.py --image_dir " + output_dir + " --output " + output_dir + "/animation.gif --fps 10";
    int python_ret = system(python_command.c_str());
    if (python_ret != 0) {
        std::cerr << "Error: Python animation script failed." << std::endl;
        return -1;
    }

    char* output_bucket_env = getenv("OUTPUT_BUCKET");
    std::string output_bucket = output_bucket_env ? std::string(output_bucket_env) : "yf23-fft2d-output";

    // Upload animation to S3
    std::string animation_path = output_dir + "/animation.gif";
    std::string animation_key = timestamp_str + "/animation.gif";
    uploadToS3(animation_path, output_bucket, animation_key);

    // Upload final reconstruction to S3
    std::string img_path = output_dir + "/reconstructed_final.png";
    std::string img_key = timestamp_str + "/reconstructed_final.png";
    uploadToS3(img_path, output_bucket, img_key);

    // Upload performance report to S3
    std::string report_path = "performance_report.txt";
    std::string report_key = timestamp_str + "/performance_report.txt";
    uploadToS3(report_path, output_bucket, report_key);

    return 0;
}
