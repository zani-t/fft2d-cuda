cmake_minimum_required(VERSION 3.10)
project(FFTProcessor LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find FFTW
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3f)
include_directories(${FFTW_INCLUDE_DIRS})
link_directories(${FFTW_LIBRARY_DIRS})

# Find CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda/bin/nvcc")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
set(CUDA_CUDART_LIBRARY "/usr/local/cuda/lib64/libcudart.so")
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Include directories
include_directories(${PROJECT_SOURCE_DIR})

# Create library target
add_library(fft_library
    image_processing.cpp
    fft2d_cuda.cu
    # Add other source files if necessary
)

# Specify include directories for the library
target_include_directories(fft_library PUBLIC ${PROJECT_SOURCE_DIR})

# Link libraries to the library target
target_link_libraries(fft_library PUBLIC ${OpenCV_LIBS} ${FFTW_LIBRARIES} pthread cufft)

# Add executable
cuda_add_executable(fft_processor main.cpp)

# Link libraries
target_link_libraries(fft_processor fft_library pthread)

# Enable testing
enable_testing()

# Add subdirectory for tests
add_subdirectory(tests)
