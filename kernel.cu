#include "kernel.h"
dim3 dimBlock(16,8);
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

void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output, bool reverse, bool norm){;}
void gspmmve(graph_t& graph, array2d_t<float>& input1, array1d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse){;}
void gspmme(graph_t& graph, array1d_t<float>& edge_input, array1d_t<float>& output, op_t op, bool reverse){;}
void gspmme2d(graph_t& graph, array2d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse){;}
void gspmmve2d(graph_t& graph, array3d_t<float>& input1, array2d_t<float>& edge_input, array3d_t<float>& output, op_t op, bool reverse){;}
void gsddmmve(graph_t& graph, array1d_t<float>& input_left, array1d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse){;}
void gsddmmve2d(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse){;}
void gsddmmvv(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse){;}
void gsddmmvv2d(graph_t& graph, array3d_t<float>& input_left, array3d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse){;}
void test_2out(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse){;}
void test3(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse){;}
void test4(array3d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, int t){;}
void Multi_plication(array2d_t<float>& image,array2d_t<float>& weight,array1d_t<float>& bias,array2d_t<float>& output,int size){;}
void Trans_pose(array2d_t<float>& input,array2d_t<float>& output,int size)
{
    dim3 dimGrid((input.col_count + dimBlock.x - 1) / dimBlock.x, (input.row_count + dimBlock.y - 1) / dimBlock.y, 1);
    matrixTranspose<<<dimGrid, dimBlock>>>(input.data_ptr, output.data_ptr, input.row_count, input.col_count);
    cudaDeviceSynchronize();
}
void Matrix_multiplication(array2d_t<float>& image,array2d_t<float>& weight,array2d_t<float>& output, int size)
{    
    dim3 dimGrid((weight.col_count + dimBlock.x - 1) / dimBlock.x, (image.row_count + dimBlock.y - 1) / dimBlock.y, 1);
    
    matrixMul2D<<<dimGrid, dimBlock>>>(image.data_ptr, weight.data_ptr, output.data_ptr, image.row_count, image.col_count, weight.col_count);
    cudaDeviceSynchronize();
}