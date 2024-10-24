#include <opencv2/opencv.hpp>
#include <algorithm>
#include <string>
#include <iostream>

// Function to round down to the nearest power of 2
int round_down_to_power_of_2(int n) {
    if (n < 1) return 0; // If the number is less than 1, return 0 (no power of 2)
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    return n - (n >> 1);
}

// Function to load and crop the image
bool loadAndCropImage(const std::string& filename, cv::Mat& image, int& width, int& height) {
    // Load the image in color
    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << filename << ". Please check the file path and format." << std::endl;
        return false;
    }

    // Determine the side length
    int side_length = round_down_to_power_of_2(std::min(img.cols, img.rows));
    if (side_length < 1) {
        std::cerr << "Error: Invalid image dimensions." << std::endl;
        return false;
    }

    // Calculate top-left corner for cropping
    int x = (img.cols - side_length) / 2;
    int y = (img.rows - side_length) / 2;

    // Handle cases where cropping dimensions are invalid
    if (x < 0 || y < 0) {
        std::cerr << "Error: Cropping dimensions are invalid." << std::endl;
        return false;
    }

    // Define the region of interest
    cv::Rect roi(x, y, side_length, side_length);

    // Check if ROI is within image bounds
    if (x + side_length > img.cols || y + side_length > img.rows) {
        std::cerr << "Error: ROI exceeds image bounds." << std::endl;
        return false;
    }

    // Crop the image
    image = img(roi).clone();

    // Set width and height
    width = side_length;
    height = side_length;

    return true;
}
