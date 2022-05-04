#include <stdio.h>
#include <stdlib.h>
#include "utils/pngio.h"
#include <iostream>
#include <cmath>
#include <omp.h>

#include "/usr/local/cuda/include/cuda_runtime.h"

#define WIDTH   (256u)
#define HEIGHT  (256u)

#define BLOCK_SIZE   (16u)

double h_diag_len = 0;
__constant__ double d_diag_len;

__global__ void createImage(unsigned char * img){
    unsigned int x,y; 
    x = threadIdx.x +blockIdx.x * blockDim.x;
    y = threadIdx.y + blockIdx.y* blockDim.y;
    if((x < WIDTH) && (y<HEIGHT)){
        unsigned int i = (y* WIDTH + x )*3;
        if(y>=HEIGHT/2){
            img[i] = float(x)/WIDTH*255;
            img[i+1] = float(y)/HEIGHT*255;
            img[i+2] = sqrtf(powf(x,2)+powf(y,2))/d_diag_len *255;
        }
        else{
            img[i] = sqrtf(powf(x,2)+powf(y,2))/d_diag_len *255;
            img[i+1] =0;
            img[i+2] = 0;
        }
    }
}


#define CUDA_CHECK_RETURN( value ) {    \
    cudaError_t err = value;            \
    if(err!= cudaSuccess){               \
        fprintf(stderr,"Error %s at line %d in file %s \n",cudaGetErrorString(err),__LINE__,__FILE__);\
        exit(1);\
    }\
}

using namespace std;

int main(){
    cout<<"Creating image, please wait..."<<endl;

    int size = WIDTH*HEIGHT*3*sizeof(unsigned char);
    h_diag_len = sqrtf(powf(WIDTH,2)+powf(HEIGHT,2));

    /*Copy constant to GPU*/
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_diag_len,&h_diag_len,sizeof(double)));

    /*Allocate memory on CPU*/
    unsigned char * h_img = new unsigned char [size];

    /*Allocate memory on GPU*/
    unsigned char * d_img;
    CUDA_CHECK_RETURN(cudaMalloc(&d_img,size));

    /*Settings of kernel*/
    dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
    dim3 gridSize((WIDTH+BLOCK_SIZE-1)/BLOCK_SIZE,(HEIGHT+BLOCK_SIZE-1)/BLOCK_SIZE);

    /**/
    createImage<<<gridSize,blockSize>>>(d_img);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    /*Copy img back to CPU*/
    CUDA_CHECK_RETURN(cudaMemcpy(h_img,d_img,size,cudaMemcpyDeviceToHost));

    /*Convert data to PNG*/

    png::image<png::rgb_pixel> img(WIDTH,HEIGHT);
    pvg::rgbToPng(img,h_img);
    img.write("../test.png");
    delete[] h_img;
    CUDA_CHECK_RETURN(cudaFree(d_img));

    cout<<"Done"<<endl;

}