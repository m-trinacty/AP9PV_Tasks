#include <stdio.h>
#include <stdlib.h>
#include "utils/pngio.h"
#include <iostream>

#include "/usr/local/cuda/include/cuda_runtime.h"

#define BLOCK_SIZE (16u)

#define FILTER_SIZE  (3u)
#define TILE_SIZE  (14u) //BLOCK_SIZE - 2*(FILTER_SIZE/2)

#define CUDA_CHECK_RETURN( value ) {    \
    cudaError_t err = value;            \
    if(err!= cudaSuccess){               \
        fprintf(stderr,"Error %s at line %d in file %s \n",cudaGetErrorString(err),__LINE__,__FILE__);\
        exit(1);\
    }\
}

__device__ int edgeFilter [3][3] = {{1,0,-1},{0,0,0},{-1,0,1}}; 
__device__ int topSobelFilter [3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

__device__ int customFilter [3][3] = {{1,2,-1},{-2,0,-1},{0,2,0}}; 
__global__ void processImag(unsigned char * out,unsigned char * in,size_t pitch, unsigned int width, unsigned int height, int filter ){
    int x_o = TILE_SIZE*blockIdx.x+threadIdx.x;
    int y_o = TILE_SIZE*blockIdx.y+threadIdx.y;
    int x_i = x_o-2;
    int y_i = y_o-2;
    int sum=0;
    __shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];
    if(( x_i >= 0 ) && (x_i < width) &&(y_i >= 0) && (y_i < height)){
        
        sBuffer[threadIdx.y][threadIdx.x]= in[y_i*pitch+x_i];
    }
    else{
        sBuffer[threadIdx.y][threadIdx.x]= 0;
    }
    __syncthreads();

    if(threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE){
        for(int r=0;r<FILTER_SIZE; r++){
            for(int c =0;c<FILTER_SIZE;c++){
            switch(filter){
                case 0:
                    sum+=sBuffer[threadIdx.y+r][threadIdx.x+c]*(edgeFilter[r][c]);
                    
                break;
                case 1:
                    sum+=sBuffer[threadIdx.y+r][threadIdx.x+c]*(topSobelFilter[r][c]);

                break;
                case 2:                    
                    sum+=sBuffer[threadIdx.y+r][threadIdx.x+c]*(customFilter[r][c]);
                break;
            }
            }
        }
        //sum/= FILTER_SIZE*FILTER_SIZE;
        if(x_o<width && y_o<height){
            out[y_o*width+x_o]=sum;
        }
    }

    
}
int main(){
    png::image<png::rgb_pixel> img("../lena.png");

    std::cout << "**********************************"<<std::endl;
    std::cout << "*Image filter on TeslaGPU        *"<<std::endl;    
    std::cout << "*                                *"<<std::endl;
    std::cout << "*Author: Martin Trinacty         *"<<std::endl;
    std::cout << "**********************************"<<std::endl;    
    int selectedFilter=-1;
    while(selectedFilter<0||selectedFilter>2)
    {
    std::cout << "**********************************"<<std::endl;
    std::cout << "*Please select filter            *"<<std::endl;    
    std::cout << "*1>>Edge Filter                  *"<<std::endl;  
    std::cout << "*2>>TopSobel Filter              *"<<std::endl;  
    std::cout << "*3>>Custom Filter                *"<<std::endl;
    std::cout << "**********************************"<<std::endl;    
    std::cin>> selectedFilter;
    selectedFilter--;
    if(selectedFilter<0||selectedFilter>2){
        std::cout << "**********************************"<<std::endl;
        std::cout << "*Wrong selection                 *"<<std::endl;   
        std::cout << "*Try again                       *"<<std::endl;
        std::cout << "**********************************"<<std::endl;   
    }
    }

    unsigned int width= img.get_width();
    unsigned int height= img.get_height();

    int size = width*height*sizeof(unsigned char);

    unsigned char *h_r = new unsigned char [size];
    unsigned char *h_g = new unsigned char [size];
    unsigned char *h_b = new unsigned char [size];
    
    unsigned char *h_r_n = new unsigned char [size];
    unsigned char *h_g_n = new unsigned char [size];
    unsigned char *h_b_n = new unsigned char [size];
    pvg::pngToRgb3(h_r,h_g,h_b,img);

    unsigned char *d_r = NULL;
    unsigned char *d_g = NULL;
    unsigned char *d_b = NULL;

    size_t pitch_r = 0;
    size_t pitch_g = 0;
    size_t pitch_b = 0;
    
    unsigned char *d_r_n = NULL;
    unsigned char *d_g_n = NULL;
    unsigned char *d_b_n = NULL;
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_r,&pitch_r,width,height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_g,&pitch_g,width,height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_b,&pitch_b,width,height));

    CUDA_CHECK_RETURN(cudaMalloc(&d_r_n,size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_n,size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_n,size));

    CUDA_CHECK_RETURN(cudaMemcpy2D(d_r,pitch_r,h_r,width,width,height,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_g,pitch_g,h_g,width,width,height,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_b,pitch_b,h_b,width,width,height,cudaMemcpyHostToDevice));

    dim3 gridSize((width+ TILE_SIZE -1)/TILE_SIZE,(height+ TILE_SIZE -1)/TILE_SIZE);
    dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
    processImag<<<gridSize,blockSize>>>(d_r_n,d_r,pitch_r,width,height,selectedFilter);
    processImag<<<gridSize,blockSize>>>(d_g_n,d_g,pitch_g,width,height,selectedFilter);
    processImag<<<gridSize,blockSize>>>(d_b_n,d_b,pitch_b,width,height,selectedFilter);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaMemcpy(h_r_n,d_r_n,size,cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_g_n,d_g_n,size,cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_b_n,d_b_n,size,cudaMemcpyDeviceToHost));

    pvg::rgb3ToPng(img,h_r_n,h_g_n,h_b_n);
    img.write("../lenaNew.png");
    CUDA_CHECK_RETURN(cudaFree(d_r));
    CUDA_CHECK_RETURN(cudaFree(d_g));
    CUDA_CHECK_RETURN(cudaFree(d_b));
    CUDA_CHECK_RETURN(cudaFree(d_r_n));
    CUDA_CHECK_RETURN(cudaFree(d_g_n));
    CUDA_CHECK_RETURN(cudaFree(d_b_n));

    delete [] h_r;
    delete [] h_g;
    delete [] h_b;
    delete [] h_r_n;
    delete [] h_g_n;
    delete [] h_b_n;


    std::cout << "**********************************"<<std::endl;
    std::cout << "*Your image is done              *"<<std::endl;   
    std::cout << "*                                *"<<std::endl;
    std::cout << "**********************************"<<std::endl;   

}