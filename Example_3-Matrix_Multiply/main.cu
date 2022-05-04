#include <stdio.h>
#include <stdlib.h>
#include "utils/pngio.h"
#include <iostream>
#include <cmath>
#include <omp.h>

#include "/usr/local/cuda/include/cuda_runtime.h"

#define WIDTH   (800u)
#define HEIGHT  (600u)

#define MATRIX_SIZE (4u)


#define A_WIDTH (2u)
#define A_HEIGHT (2u)

#define B_WIDTH (2u)
#define B_HEIGHT (2u)
#define BLOCK_SIZE   ((MATRIX_SIZE/2))

double h_diag_len = 0;
__constant__ double d_diag_len;

__global__ void createImage(unsigned char * img){
    unsigned int x,y;
    x = threadIdx.x +blockIdx.x * blockDim.x;
    y = threadIdx.y + blockIdx.y* blockDim.y;
    if((x < WIDTH) && (y<HEIGHT)){
        unsigned int i = (y* WIDTH + x )*3;
        img[i] = float(x)/WIDTH*255;
        img[i+1] = float(y)/HEIGHT*255;
        img[i+2] = sqrtf(powf(x,2)+powf(y,2))/d_diag_len *255;
    }
}
__global__ void Add(int* m1, int * m2, int * res){
    unsigned int x,y;
    x = threadIdx.x +blockIdx.x * blockDim.x;
    y = threadIdx.y + blockIdx.y* blockDim.y;
    if((x < WIDTH) && (y<HEIGHT)){
        unsigned int i = ( y * MATRIX_SIZE + x )*3;
        res[i] = m1[i]+m2[i]; 
    }
}

__global__ void Multiply(int* m1, int * m2, int * res){
    unsigned int col,row;
    int val=0;
    col = blockIdx.x*blockDim.x+threadIdx.x;
    row = blockIdx.y*blockDim.y+threadIdx.y;
    if((col <B_HEIGHT) && (row<A_WIDTH)){
        for(int i= 0;i<A_WIDTH;++i){
            val+= m1[row*A_WIDTH+i]*m2[i*B_WIDTH+col];
        } 
    }
    res[row*B_WIDTH+col]=val;
}
#define CUDA_CHECK_RETURN( value ) {    \
    cudaError_t err = value;            \
    if(err!= cudaSuccess){               \
        fprintf(stderr,"Error %s at line %d in file %s \n",cudaGetErrorString(err),__LINE__,__FILE__);\
        exit(1);\
    }\
}

using namespace std;
void fillTestMatrix(int * arr){
    
    for(int i=0;i<MATRIX_SIZE*MATRIX_SIZE;i++){
          arr[i]=(i%3)+1;
    }
    
}
int main(){
    cout<<"Calculating matrix"<<endl;

    //int size = MATRIX_SIZE*MATRIX_SIZE*sizeof(int);
    int sizeM1 = A_WIDTH*A_HEIGHT*sizeof(int);
    int sizeM2 = B_WIDTH*B_HEIGHT*sizeof(int);
    int sizeRes = sizeM1==sizeM2? sizeM1:sizeM2;
    /*Copy constant to GPU*/
    //CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_diag_len,&h_diag_len,sizeof(double)));

    /*Allocate memory on CPU*/
    int * h_matrix1= new int[9];
    int test1[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    h_matrix1= test1;
    int * h_matrix2= new int[6];
    int test2[6] = {8, 9, -5, 4, 10, -1};
    h_matrix2= test2;
    int * h_res= new int[6];

    /*Allocate memory on GPU*/
    int * d_matrix1;
    int * d_matrix2;
    int * d_res;
    CUDA_CHECK_RETURN(cudaMalloc(&d_matrix1,sizeM1));
    CUDA_CHECK_RETURN(cudaMalloc(&d_matrix2,sizeM2));
    CUDA_CHECK_RETURN(cudaMalloc(&d_res,sizeRes));
    /*Copy array to device*/
    CUDA_CHECK_RETURN(cudaMemcpy(d_matrix1,h_matrix1,sizeM1,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_matrix2,h_matrix2,sizeM2,cudaMemcpyHostToDevice));


    /*Settings of kernel*/
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 gridSize((MATRIX_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE,(MATRIX_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE); 
    
    dim3 gridSizeMul((A_WIDTH*A_HEIGHT)/BLOCK_SIZE, (B_WIDTH*B_HEIGHT)/BLOCK_SIZE);

    /**/
    Multiply<<<gridSizeMul,blockSize>>>(d_matrix1,d_matrix2,d_res);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    /*Copy array back to CPU*/
    CUDA_CHECK_RETURN(cudaMemcpy(h_res,d_res,sizeRes,cudaMemcpyDeviceToHost));


    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    for(int i=0;i<(sizeM1/sizeof(int));i++){
        cout<<h_matrix1[i]<<" ";
        if(i%A_WIDTH==A_WIDTH-1 )
            cout<<endl;
    }
    cout<<endl;
    cout<<endl;
    for(int i=0;i<(sizeM2/sizeof(int));i++){
        cout<<h_matrix2[i]<<" ";
        if(i%B_WIDTH==B_WIDTH-1)
            cout<<endl;
    }
    cout<<endl;
    cout<<endl;
    for(int i=0;i<(sizeRes/sizeof(int));i++){
        cout<<h_res[i]<<" ";
        if(sizeM1==sizeM2 ){
            if(i%A_WIDTH==A_WIDTH-1)
            cout<<endl;
        }
        else{
            if(i%B_WIDTH==B_WIDTH-1)
            cout<<endl;
        }
    }
    
    cout<<endl;
    delete[] h_matrix1;
    delete[] h_matrix2;

    delete[] h_res;
    CUDA_CHECK_RETURN(cudaFree(d_matrix1));
    CUDA_CHECK_RETURN(cudaFree(d_matrix2));
    CUDA_CHECK_RETURN(cudaFree(d_res));

    cout<<"Done"<<endl;

}