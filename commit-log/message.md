# Using the GPU for image processing with CNN

## Summary
The implementation added in this commit moves some of the processing work to the Nvidia GPU, where parallelism is used to break down a large matrix and conduct the same operation on different sub matrices. Overall, two files were largely modified: `<kernel.cu>` (the CUDA-C code that tells the GPU what to do) and `<cuda.rs>` (the Rust code that prepares the work and launches the CUDA-C code).

## Tech details

### `<cuda.rs>`: preparing and initiating the GPU tasks
The Rust side of this implementation includes two main functions: `init()` and `compute()` inside the `CudaContext` struct. The `init()` function takes the convolution layer and output layer from a given CNN and initializes a group of variables that will be needed when launching the kernel (such as instances of `Module`, `Stream`, and `Context`). The `compute()` function takes in a single input matrix (an image) and launches the kernel multiple times by passing the input matrix sequentially through the convolution layer, ReLU layer, and output layer.

One thing that needs to be paid particular attention to was the `<<<...>>>` block in each `launch!()`, which specifies the number of blocks and the number of thread in each block (which has to be a multiple of 32). Based on the need of threads by `<kernel.cu>` (in the section below),
- `<<<10, 512, 0, stream>>>` was used to launch the convolution layer and ReLU layer, and
- `<<<1, 32, 0, stream>>>` was used to launch the output layer (since 32 is the smallest multiple of 32 that is greater than or equal to 10).

### `<kernel.cu>`: the kernel code written for the GPU
The kernel code in `<kernel.cu>` is what the GPU actually runs. It utilizes `blockIdx` and `threadIdx` to achieve parallelism. Instead of using for loops, this kernel file lets multiple threads run the same instruction simultaneously. For example, in the `filter_w_conv_layer()` function, each thread (with a unique `blockIdx.x` and `threadIdx.x`) computes a single value in the 3D matrix `layer1_output`. Then, by using (10\*20\*20=4000) threads, 4000 values in the 3D matrix `layer1_output` can be computed all at once. If `threadIdx.x` or `blockIdx.x` goes out of bounds, this function will terminate without doing further work. It is also worth mentioning that for higher efficiency, the ReLU layer is also included in this function together with the convolution layer.

- On the convolution layer and ReLU layer (in `filter_w_conv_layer()`), 10 blocks of threads were needed, each of which contains 400 threads.
- On the output layer (in `filter_w_output_layer()`), a single block of 10 threads were used.

### Other notes
Since Nvidia GPUs are needed to run the CUDA-C code, it is important to note that the program should be run specifically on the `eceTesla` machines. For this assignment, when testing the program for its correctness and performance, the server used was `eceTesla1`.

Some issues were encountered when `<cuda.rs>` was being written, and modifications had to be made to fix these unexpected issues. For example, instead of passing `self.module` and `self.stream` into the `launch!()` function, RustaCUDA somehow required the plain `module` and `stream` to be passed in, meaning something like `let module = &self.module;` was needed before calling `launch!()`.


## Testing for correctness
To test for correctness, the following two commands were run:
- `cargo run --release -- cpu input/cnn.csv input/in.csv output/out.csv`
- `cargo run --release -- cuda input/cnn.csv input/in.csv output/out_cuda.csv`
which updates `out.csv` and `out_cuda.csv` respectively with the program outputs. Then, the Python script `<compare.py>` was run, and the GPU outputs were proven to be correct since "Comparison finished" was printed in the terminal.

To further verify the correctness, the `<generate.py>` script was also run 3 times to generate different input matrices for more tests. In all the 3 experiements, the outputs were tested to be the same as the CPU implementation.

## Testing for performance
To test for performance, the period spent by the program was recorded by observing the "... microseconds of actual work done" line in the program output. After running the CPU and GPU implementations 3 times,
- the normal CPU approach took 60350ms, 63192ms, and 59835ms, and
- the GPU approach took 114215ms, 115643ms, and 117442ms.

which was unexpected since the GPU approach was around 2 times slower than the CPU approach, so further investigations were conducted on the 2 `launch!()` operations of the kernel: one for the convolution layer computation, one for the output layer computation.

In the investigation, the performance of the convolution and ReLU layers was first tested in the CPU and GPU implementation (by commenting out the code that runs the output layer). On the first 2 layers, the CPU spent around 26000ms whereas the GPU spent around 12500ms (more than 2 times faster), meaning the GPU approach performs better than the CPU approach on the convolution layer and ReLU layer. Therefore, the output layer computation is what caused the GPU to consume more time than the CPU. A reasonable guess is that the output layer is "not worth" the effort of launching the kernel to the GPU for computation, and thus the plain CPU approach is faster in this case.


