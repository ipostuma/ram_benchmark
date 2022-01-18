# ram_benchmark
A python script to benchmark memory usage for 3D Unet models trained with Keras and TF

## memorybenchmark.py

The scripts generates a simple 3D Unet model with random data (for the scope of this work real images are not necessary). 
The 3D images have a single input channel and a single output channel. This is an oversempification wich will be addressed in the future.

## loop.sh

The scripts iterates though images shapes (64, 128, 256, 512) and model depth (3, 4, 5, 6, 7). Tensorboar profile evaluations are saved in logs/fit.
