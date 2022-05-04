#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <cstdlib>

#include "/usr/local/cuda/include/cuda_runtime.h"
#define MANAGED_MEMORY (false)
#define VECT_SIZE (4096u) 
//#define BLOCK_SIZE (128u)

#define CUDA_CHECK_RETURN( value ) {    \
    cudaError_t err = value;            \
    if(err!= cudaSuccess){               \
        fprintf(stderr,"Error %s at line %d in file %s \n",cudaGetErrorString(err),__LINE__,__FILE__);\
        exit(1);\
    }\
}


__global__ void fillVector(int *data){
    int i = threadIdx.x +blockIdx.x * blockDim.x;
    if(i<VECT_SIZE){
            data[i] = i+1;
    }
}
__global__ void fillVectorSubtract(int *data, int subtract){
    int i = threadIdx.x +blockIdx.x * blockDim.x;
    if(i<VECT_SIZE){
            data[i] = i-subtract;
    }
}

__global__ void subtractVectors(int *vector1, int * vector2, int * result){
    int i = threadIdx.x +blockIdx.x * blockDim.x;
    if(i<VECT_SIZE){
            result[i] = vector2[i]-vector1[i];
    }
}

int main(){
    
    #if MANAGED_MEMORY
    std::cout<<"MANAGED MEMORY"<<std::endl;
    int *data = NULL;
    int *data2 = NULL;

    int *res = NULL; 
    
    CUDA_CHECK_RETURN(cudaMallocManaged(&data,VECT_SIZE * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMallocManaged(&data2,VECT_SIZE * sizeof(int)));
    
    CUDA_CHECK_RETURN(cudaMallocManaged(&res,VECT_SIZE * sizeof(int)));
    #else
    std::cout<<"UNMANAGED MEMORY"<<std::endl;
    //allocation memory on CPU
    int * h_data =(int*)malloc(VECT_SIZE *sizeof(int));
    memset(h_data,0,VECT_SIZE*sizeof(int));
    int * h_data2 =(int*)malloc(VECT_SIZE *sizeof(int));
    int * h_res =(int*)malloc(VECT_SIZE *sizeof(int));


    //allocation memory on GPU
    int * d_data = NULL;
    CUDA_CHECK_RETURN(cudaMalloc(&d_data,VECT_SIZE * sizeof(int)));
    int * d_data2 = NULL;
    CUDA_CHECK_RETURN(cudaMalloc(&d_data2,VECT_SIZE * sizeof(int)));
    int * d_res = NULL;
    CUDA_CHECK_RETURN(cudaMalloc(&d_res,VECT_SIZE * sizeof(int)));

    #endif
    /*kernel configuration*/
    int blockSize = BLOCK_SIZE;
    int gridSize = (VECT_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;


    #if MANAGED_MEMORY
    for(int i =0;i<VECT_SIZE;++i){
        data[i]=(rand()%VECT_SIZE)+1;
    }
    fillVectorSubtract<<<gridSize,blockSize>>>(data2,5);    
    #else
    for(int i =0;i<VECT_SIZE;++i){
        h_data[i]=(rand()%VECT_SIZE)+1;
    }
    cudaMemcpy(d_data,h_data,VECT_SIZE*sizeof(int),cudaMemcpyHostToDevice);
    fillVectorSubtract<<<gridSize,blockSize>>>(d_data2,5);
    #endif

    //Wait for filled vector before substraction
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    #if MANAGED_MEMORY
    subtractVectors<<<gridSize,blockSize>>>(data,data2,res);
    #else
    subtractVectors<<<gridSize,blockSize>>>(d_data,d_data2,d_res);
    #endif
   
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    #if !MANAGED_MEMORY
    /*copy data back to CPU*/
    CUDA_CHECK_RETURN(cudaMemcpy(h_data2,d_data2,VECT_SIZE*sizeof(int),cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaMemcpy(h_res,d_res,VECT_SIZE*sizeof(int),cudaMemcpyDeviceToHost));
    /*no need to copy h_data, it was initialized on host */
    #endif
       
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    int printSize = VECT_SIZE>32?32:VECT_SIZE;
    if(VECT_SIZE>printSize){
        std::cout<<"Printing only first 32 computations out of "<<VECT_SIZE<<std::endl;
    }
    std::cout<<std::endl;
    std::cout <<"Computation"<<std::endl;
    for(int i =0 ;i< printSize;++i){
        #if MANAGED_MEMORY
        std::cout<< data2[i]<<"-"<<data[i]<<"="<<res[i]<<std::endl;
        #else
        std::cout<< h_data2[i]<<"-"<<h_data[i]<<"="<<h_res[i]<<std::endl;
        #endif
        
    }

    std::cout<<std::endl;
    std::cout <<"Data2 in host"<<std::endl;
    for(int i =0 ;i< printSize;++i){
        #if MANAGED_MEMORY
        std::cout<< data2[i];
        #else
        std::cout<< h_data2[i];
        #endif
        if(i!=VECT_SIZE-1){
            std::cout<<", ";
        }
    }

    std::cout<<std::endl;
    std::cout <<"Data in host"<<std::endl;
    for(int i =0 ;i< printSize;++i){
        #if MANAGED_MEMORY
        std::cout<< data[i];
        #else
        std::cout<< h_data[i];
        #endif
        if(i!=VECT_SIZE-1){
            std::cout<<", ";
        }
    }

    
    
    std::cout<<std::endl;
    std::cout <<"Result in host"<<std::endl;
    for(int i =0 ;i< printSize;++i){
        #if MANAGED_MEMORY
        std::cout<< res[i];
        #else
        std::cout<< h_res[i];
        #endif
        if(i!=VECT_SIZE-1){
            std::cout<<", ";
        }
    }
    std::cout<<std::endl;
    
    
    //free memory
    #if MANAGED_MEMORY
    CUDA_CHECK_RETURN(cudaFree(data));
    CUDA_CHECK_RETURN(cudaFree(data2));
    CUDA_CHECK_RETURN(cudaFree(res));
    #else
    free(h_data);
    free(h_data2);
    free(h_res);
    CUDA_CHECK_RETURN(cudaFree(d_data));
    CUDA_CHECK_RETURN(cudaFree(d_data2));
    CUDA_CHECK_RETURN(cudaFree(d_res));
    #endif
}