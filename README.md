# 2d-Convolution-CUDA
This is a simple 2d convolution written in cuda c which uses shared memory for better performance

Curerntly used the block size as 32 and image dimensions 512 x 512 with kernel dimension 3 x 3

Do update the block size also while changing the image dimensions to see if enough threads are allocated.
