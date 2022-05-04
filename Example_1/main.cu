#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "/usr/local/cuda/include/cuda_runtime.h"

#define CUDA_CHECK_RETURN( value ) {    \
    cudaError_t err = value;            \
    if(err!= cudaSuccess){               \
        fprintf(stderr,"Error %s at line %d in file %s \n",cudaGetErrorString(err),__LINE__,__FILE__);\
        exit(1);\
    }\
}

int main(){
    int devCount;
    CUDA_CHECK_RETURN( cudaGetDeviceCount(&devCount) );
    cudaDeviceProp properties;
    for(int i = 0;i<devCount; ++i){
        CUDA_CHECK_RETURN(cudaGetDeviceProperties(&properties,i));
        std::cout << "Device "<< i << " name: "<<properties.name<<std::endl;
        std::cout << "Compute compability:  " << properties.major << "." << properties.minor << std::endl;
        std::cout << "Block dimensions: " << properties.maxThreadsDim[0]<<","<< properties.maxThreadsDim[1]<<","<<properties.maxThreadsDim[2]<<std::endl;

        std::cout << "Grid dimmensions: " << properties.maxGridSize[0]<<","<< properties.maxGridSize[1] <<","<<properties.maxGridSize[2]<<std::endl;

    }
    printf("Found %i CUDA graphics Card\n", devCount);
    printf("Hello world!\n");
}