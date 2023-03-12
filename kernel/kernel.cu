// Very minimal skeleton for the kernel

#include <stdio.h>

extern "C" __global__ void filter_w_output_layer(
    double output_layer_input[10][20][20],
    double output_layer[10][4000],
    double output_layer_output[10]
) {
    if (blockIdx.x >= 20 || threadIdx.x >= 20) {
        return;
    }
    for (int fNeuron=0; fNeuron<10; fNeuron++) {
        for (int fInput=0; fInput<10; fInput++) {
            output_layer_output[fNeuron] += output_layer_input[fInput][blockIdx.x][threadIdx.x] * output_layer[fNeuron][fInput*20*20 + blockIdx.x*20 + threadIdx.x];
        }
    }
}

extern "C" __global__ void filter_w_relu_layer(
    double layer1_output[10][20][20]
){
    if (blockIdx.x >= 20 || threadIdx.x >= 20) {
        return;
    }
    for (int i=0; i<10; i++) {
        if (layer1_output[i][blockIdx.x][threadIdx.x] < 0.0) {
            layer1_output[i][blockIdx.x][threadIdx.x] = 0.0;
        }
    }
}

extern "C" __global__ void filter_w_conv_layer(
    double input_matrix[100][100],
    double conv_layer[10][5][5],
    double layer1_output[10][20][20]
) {
    if (blockIdx.x >= 100 || threadIdx.x >= 100) {
        return;
    }
    for (int f=0; f<10; f++) {
        // Find which sub matrix we are in (W.R.T. the input matrix):
        int matrixRowIdx = (blockIdx.x)/5;
        int matrixColumnIdx = (threadIdx.x)/5;
        layer1_output[f][matrixRowIdx][matrixColumnIdx] += input_matrix[blockIdx.x][threadIdx.x] * conv_layer[f][(blockIdx.x)%5][(threadIdx.x)%5];
    }

    /*
    // The C++ version of convolution_layer() in <cpu.rs>:
    for (int f = 0; f < conv_layer.size(); ++f) {
        vector<vector<double>> filter = conv_layer[f];
        vector<vector<double>> out = layer1_output[f];
        for (int i = 0; i < 100; i += 5) {
            for (int j = 0; j < 100; j += 5) {
                double prod = 0.0;
                for (int x = 0; x < 5; ++x) {
                    for (int y = 0; y < 5; ++y) {
                        prod += input_matrix[i + x][j + y] * filter[x][y];
                    }
                }
                out[i / 5][j / 5] = prod;
            }
        }
    }
    */
}