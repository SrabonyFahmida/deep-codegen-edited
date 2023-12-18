#include "kernel.h"
#define TILE_DIM 32  // Define the tile size
//#define TILE_SIZE 16  // Block dimension that divides 28
//dim3 dimBlock(32,32);
//dim3 dimGrid(14,14);
dim3 dimBlock(32,32);
__global__ void matrixMul2D(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int i = 0; i < colsA; ++i) {
            sum += A[row * colsA + i] * B[i * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}
__global__ void matrixTranspose(float *A, float *B, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numRows && col < numCols) {
        B[col * numRows + row] = A[row * numCols + col];
    }
}


__global__ void matrixMulCoalesced(float* A, float* B, float* C, int M, int N, int K) {
    //32 is TILE_DIM

    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Calculate the row index and column index in the C matrix
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum = 0.0f;

    // Loop over the tiles of the matrices
    for (int m = 0; m < (K + 32 - 1) / 32; ++m) {
        // Load the current tile of A and B into shared memory
        if (m * 32 + tx < K && row < M) {
            tileA[ty][tx] = A[row * K + m * 32 + tx];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        if (m * 32+ ty < K && col < N) {
            tileB[ty][tx] = B[(m * 32 + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < 32; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write the result to the C matrix
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}






void Trans_pose(array2d_t<float>& input,array2d_t<float>& output){
    dim3 dimGrid((input.col_count + dimBlock.x - 1) / dimBlock.x, (input.row_count + dimBlock.y - 1) / dimBlock.y, 1);
    matrixTranspose<<<dimGrid, dimBlock>>>(input.data_ptr, output.data_ptr, input.row_count, input.col_count);
    cudaDeviceSynchronize();
    }
void Matrix_multiplication(array2d_t<float>& image,array2d_t<float>& weight,array2d_t<float>& output){
    dim3 dimGrid((weight.col_count + dimBlock.x - 1) / dimBlock.x, (image.row_count + dimBlock.y - 1) / dimBlock.y, 1);
    
    matrixMul2D<<<dimGrid, dimBlock>>>(image.data_ptr, weight.data_ptr, output.data_ptr, image.row_count, image.col_count, weight.col_count);
    cudaDeviceSynchronize();
    }


void Matrix_multiplication_Coal(array2d_t<float>& input1,array2d_t<float>& input2,array2d_t<float>& output){
      dim3 dimGrid((input2.col_count + dimBlock.x - 1) / dimBlock.x, (input1.row_count + dimBlock.y - 1) / dimBlock.y, 1);
    
    matrixMulCoalesced<<<dimGrid, dimBlock>>>(input1.data_ptr, input2.data_ptr, output.data_ptr, input1.row_count, input2.col_count, input2.row_count);
    cudaDeviceSynchronize(); 
    }
