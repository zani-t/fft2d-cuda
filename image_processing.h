#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <string>

// Function to load and crop the image
bool loadAndCropImage(const std::string& filename, cv::Mat& image, int& width, int& height);

#endif // IMAGE_PROCESSING_H
