// Very minimal skeleton for the kernel

#include <stdio.h>

extern "C" __global__ void filter_w_output_layer(
    double output_layer_input[10][20][20],
    double output_layer[10][4000],
    double output_layer_output[10]
) {
    /*
        threadIdx.x: [0, 9]
    */
    if (threadIdx.x >= 10) {
        return;
    }
    for (int iMatrix=0; iMatrix<10; iMatrix++) {
        for (int iRow=0; iRow<20; iRow++) {
            for (int iColumn=0; iColumn<20; iColumn++) {
                output_layer_output[threadIdx.x] += output_layer_input[iMatrix][iRow][iColumn] * output_layer[threadIdx.x][iMatrix*20*20+iRow*20+iColumn];
            }
        }
    }
}

extern "C" __global__ void filter_w_conv_layer(
    double input_matrix[100][100],
    double conv_layer[10][5][5],
    double layer1_output[10][20][20]
) {
    /*
        blockIdx.x: [0, 9], threadIdx.x: [0, 399]
    */
    if (blockIdx.x >= 10 || threadIdx.x >= 400) {
        return;
    }
    // Compute the starting row and column in `input_matrix`:
    int row = threadIdx.x / 20;
    int column = threadIdx.x % 20;
    for (int iRow=0; iRow<5; iRow++) {
        for (int iColumn=0; iColumn<5; iColumn++) {
            layer1_output[blockIdx.x][row][column] += input_matrix[row*5 + iRow][column*5 + iColumn] * conv_layer[blockIdx.x][iRow][iColumn];
        }
    }
    // Append the ReLU layer here:
    if (layer1_output[blockIdx.x][row][column] < 0) {
        layer1_output[blockIdx.x][row][column] = 0;
    }
}