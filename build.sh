#!/bin/bash

set -e

# Check if an image name is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <image_name>"
  exit 1
fi

# Input file
image_name=$1

docker build -t fft2d-processor:$image_name .
docker tag fft2d-processor:latest 924994412956.dkr.ecr.us-east-1.amazonaws.com/fft2d-processor:$image_name
docker push 924994412956.dkr.ecr.us-east-1.amazonaws.com/fft2d-processor:$image_name
