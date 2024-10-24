# Dockerfile

# Use NVIDIA's CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    cmake \
    git \
    libopencv-dev \
    python3 \
    python3-pip \
    libfftw3-dev \
    libfftw3-single3 \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip3 install imageio matplotlib opencv-python

# Install Google Test for unit testing
RUN git clone https://github.com/google/googletest.git /usr/src/googletest \
    && cd /usr/src/googletest \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Copy application code into the container
COPY . /app
WORKDIR /app

# Create build directory
RUN mkdir build && cd build && \
    cmake .. && \
    make

# Set the entrypoint to a bash script that runs the tests and main application
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# RUN compute-sanitizer --tool memcheck ./build/fft_processor

USER root

ENTRYPOINT ["./entrypoint.sh"]
