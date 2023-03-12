# Using the GPU for image processing with CNN

## Summary
The implementation added in this commit moves some of the processing work to the Nvidia GPU, where parallelism is used to break down a large matrix and conduct the same operation on different sub matrices. Overall, two files were largely modified: `kernel.cu` (the CUDA-C code that tells the GPU what to do) and `cuda.rs` (the Rust code that prepares the work and launches the CUDA-C code).

## Tech details

### Preparing the work using RustaCUDA
The Rust side of this implementation includes two main functions: `int()` and `compute()` inside the `CudaContext` struct. The `init()` function takes the convolution layer and output layer from a given CNN and initializes a group of variables that will be needed when launching the kernel (such as instances of `Module`, `Stream`, and `Context`). The `compute()` function takes in a single input matrix (an image) and launches the kernel 3 times by passing the input matrix sequentially through the convolution layer, ReLU layer, and output layer.

### Doing the work on the GPU
The code in `kernel.cu` is what the GPU actually runs. It utilizes `blockIdx` and `threadIdx` to achieve parallelism. Instead of using for loops, this kernel file lets multiple threads run the same instruction simultaneously. For example, for a 100x100 input matrix, `kernel.cu` starts 10000 threads and each thread will handle a single value in the input matrix. The number of threads used in each of the 3 layers were determined case by case, and the functions will terminate without doing further work if `threadIdx` and `blockIdx` are out of bounds.

### Using other tricks
Some issues were encountered when `cuda.rs` was being written, and modifications had to be made to fix these unexpected issues. For example, instead of passing `self.module` and `self.stream` into the `launch!()` function, RustaCUDA somehow required the plain `module` and `stream` to be passed in, meaning something like `let module = &self.module;` was needed before calling `launch!()`.


## Testing for correctness
To test for correctness, the following two commands were run:
- `cargo run --release -- cpu input/cnn.csv input/in.csv output/out.csv`
- `cargo run --release -- cuda input/cnn.csv input/in.csv output/out_cuda.csv`
which updates `out.csv` and `out_cuda.csv` respectively with the program outputs. Then, the Python script `compare.py` was run, and it would point out which line sees discrepancy in the two csv files.

## Testing for performance
To test for performance, the period spent by the program was recorded by observing the `... microseconds of actual work done` line in the program output. After running the CPU and GPU implementations 3 times, the normal CPU approach took 39106ms, 40982ms, and 56621ms, and the GPU approach took 21314ms, 25530ms, and 23967ms. Therefore, it can be concluded that the GPU implementation is consistently faster than the CPU implementation.


