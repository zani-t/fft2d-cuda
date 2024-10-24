#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting unit tests..."
cd build
ctest --output-on-failure
TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
    echo "Unit tests failed. Exiting." >&2
    exit 1
else
    echo "All unit tests passed."
fi

cd ..

# Read input S3 parameters from environment variables
INPUT_BUCKET=${INPUT_BUCKET}
INPUT_KEY=${INPUT_KEY}

if [ -z "$INPUT_BUCKET" ] || [ -z "$INPUT_KEY" ]; then
    echo "Error: INPUT_BUCKET or INPUT_KEY environment variables are not set." >&2
    exit 1
fi

echo "Downloading input image from S3: s3://$INPUT_BUCKET/$INPUT_KEY"
aws s3 cp s3://$INPUT_BUCKET/$INPUT_KEY ./input_image.jpg

if [ ! -f "input_image.jpg" ]; then
    echo "Error: Failed to download input image from S3." >&2
    exit 1
fi

echo "Running FFT processor..."
./build/fft_processor input_image.jpg output

echo "Process completed successfully."
