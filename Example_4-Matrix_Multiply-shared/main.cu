#include "utils/pngio.h"
#include <cmath>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "/usr/local/cuda/include/cuda_runtime.h"

#define MATRIX_SIZE (16u)

#define A_WIDTH (16u)
#define A_HEIGHT (16u)

#define B_WIDTH (16u)
#define B_HEIGHT (16u)
//#define BLOCK_SIZE ((16u))

double h_diag_len = 0;

__global__ void MultiplyShared(int *m1, int *m2, int *res) {
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  int *resSub = &res[A_WIDTH * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];
  int resValue = 0;
  int row = threadIdx.y;
  int col = threadIdx.x;

  for (int m = 0; m < (A_WIDTH / BLOCK_SIZE); ++m) {
    int *aSub = &m1[A_WIDTH * BLOCK_SIZE * blockRow + BLOCK_SIZE * m];
    int *bSub = &m2[B_WIDTH * BLOCK_SIZE * m + BLOCK_SIZE * blockRow];
    __shared__ int aShared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int bShared[BLOCK_SIZE][BLOCK_SIZE];
    aShared[row][col] = aSub[row * A_WIDTH + col];
    bShared[row][col] = bSub[row * B_WIDTH + col];
    __syncthreads();
    for (int e = 0; e < BLOCK_SIZE; ++e) {
      resValue += aShared[row][e] * bShared[e][col];
    }
    __syncthreads();
  }
  resSub[(row * A_WIDTH) + col] = resValue;
}
#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t err = value;                                                   \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "Error %s at line %d in file %s \n",                     \
              cudaGetErrorString(err), __LINE__, __FILE__);                    \
      exit(1);                                                                 \
    }                                                                          \
  }

using namespace std;
int main() {
  cout << "Calculating matrix" << endl;

  // int size = MATRIX_SIZE*MATRIX_SIZE*sizeof(int);
  int sizeM1 = A_WIDTH * A_HEIGHT * sizeof(int);
  int sizeM2 = B_WIDTH * B_HEIGHT * sizeof(int);
  int sizeRes = sizeM1 == sizeM2 ? sizeM1 : sizeM2;
  //int *h_matrix1 = new int[256];
  int *h_matrix1 = new int[256]{
      1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
      16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,
      31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
      46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,
      61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
      76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
      91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105,
      106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
      121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
      136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
      151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
      166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
      181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
      196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
      211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225,
      226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240,
      241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
      256};
  //h_matrix1 = test1;
  int *h_matrix2 ;
  // int test2[256] =test1;
  h_matrix2 = h_matrix1;
  int *h_res= new int [256];

  /*Allocate memory on GPU*/
  int *d_matrix1;
  int *d_matrix2;
  int *d_res;
  CUDA_CHECK_RETURN(cudaMalloc(&d_matrix1, sizeM1));
  CUDA_CHECK_RETURN(cudaMalloc(&d_matrix2, sizeM2));
  CUDA_CHECK_RETURN(cudaMalloc(&d_res, sizeRes));
  /*Copy array to device*/
  CUDA_CHECK_RETURN(
      cudaMemcpy(d_matrix1, h_matrix1, sizeM1, cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(
      cudaMemcpy(d_matrix2, h_matrix2, sizeM2, cudaMemcpyHostToDevice));

  /*Settings of kernel*/
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
  // dim3
  // gridSize((MATRIX_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE,(MATRIX_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE);

  dim3 gridSizeMul((B_WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (A_HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

  /**/
  MultiplyShared<<<gridSizeMul, blockSize>>>(d_matrix1, d_matrix2, d_res);

  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  /*Copy array back to CPU*/
  CUDA_CHECK_RETURN(cudaMemcpy(h_res, d_res, sizeRes, cudaMemcpyDeviceToHost));

  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  for (int i = 0; i < 6; i++)
    cout << h_res[i] << endl;
  for (int i = 0; i < (sizeM1 / sizeof(int)); i++) {
    cout << h_matrix1[i] << " ";
    if (i % A_WIDTH == A_WIDTH - 1)
      cout << endl;
  }
  cout << endl;
  cout << endl;
  for (int i = 0; i < (sizeM2 / sizeof(int)); i++) {
    cout << h_matrix2[i] << " ";
    if (i % B_WIDTH == B_WIDTH - 1)
      cout << endl;
  }
  cout << endl;
  cout << endl;
  for (int i = 0; i < (sizeRes / sizeof(int)); i++) {
    cout << h_res[i] << " ";
    if (sizeM1 == sizeM2) {
      if (i % A_WIDTH == A_WIDTH - 1)
        cout << endl;
    } else {
      if (i % B_WIDTH == B_WIDTH - 1)
        cout << endl;
    }
  }

  cout << endl;
  delete[] h_matrix1;
  //deleting second h_matrix2 is not allowed, because h_matrix2 is pointing to same adress asi h_matrix1, which was deleted before
  //delete[] h_matrix2;
  h_matrix2 = NULL;

  delete[] h_res;
  CUDA_CHECK_RETURN(cudaFree(d_matrix1));
  CUDA_CHECK_RETURN(cudaFree(d_matrix2));
  CUDA_CHECK_RETURN(cudaFree(d_res));

  cout << "Done" << endl;
}