# FFT2D-CUDA

Implementation of a 2D Fast Fourier Transform using CUDA parallel computing.

The code applies the FFT to the image and then iteratively reconstructs it using the inverse FFT, producing an animation of the process. It also creates a performance report comparing is computation speed to FFTW (which is of course far better).

This code was implemented on a laptop without a GPU, and was thus intended to be deployed to a GPU-enabled EC2 instance on AWS. It uses NVIDIA's CUDA Docker base image.

The workflow was as follows:
- Image uploaded to S3 input bucket
- Lambda function triggers Batch job
- Batch job starts an EC2 instance and pulls image from ECR
- Unit tests run
- Image transformed
- Frequency components of transform are iterated through, producing .gif frames of sine waves and cumulative sum
- Animation and final image saved to S3 out bucket