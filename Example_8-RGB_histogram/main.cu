#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/pngio.h"

#include "/usr/local/cuda/include/cuda_runtime.h"

//#define BLOCK_SIZE (16u)
const unsigned int Block_SIZE = 16;
const unsigned int width_img = 256;
const unsigned int height_img = 256;
#define CUDA_CHECK_RETURN( value ) {    \
    cudaError_t err = value;            \
    if(err!= cudaSuccess){               \
        fprintf(stderr,"Error %s at line %d in file %s \n",cudaGetErrorString(err),__LINE__,__FILE__);\
        exit(1);\
    }\
}
enum Colors{
    Red,
    Green,
    Blue
};

__global__ void createImage(unsigned char * channel,unsigned char * img,int color,int width,int height){
    unsigned int row = threadIdx.x +blockIdx.x * blockDim.x;
    unsigned int col = threadIdx.y + blockIdx.y* blockDim.y;
    if((row < width) && (col<height)){
        unsigned int i = (col* width + row )*3;
        if(col>=(height-channel[row])){
            switch(color){
            case Red:
                img[i] = 255;
                img[i+1] = 0;
                img[i+2] = 0;
            break;
            case Green:
                img[i] = 0;
                img[i+1] = 255;
                img[i+2] = 0;
            break;
            case Blue:
                img[i] = 0;
                img[i+1] = 0;
                img[i+2] = 255;
            break;
            }
        }
        else{
            img[i] = 0;
            img[i+1] = 0;
            img[i+2] = 0;
        }
    }
}

__global__ void histogram(unsigned int * out,unsigned char * in,size_t size,int width, int height){
    __shared__ unsigned char histo_private[Block_SIZE][Block_SIZE];
    int row = Block_SIZE * blockIdx.x+threadIdx.x;
    int col = Block_SIZE * blockIdx.y+threadIdx.y;
    if((row>= 0) && (row <width )&& (col>=0 )&&(col< height)){
        histo_private[threadIdx.y][threadIdx.x]=in[col* size +row];
    }    
    else{
        histo_private[threadIdx.y][threadIdx.x]=0;
    }
    __syncthreads();
    if(threadIdx.x <Block_SIZE && threadIdx.y<Block_SIZE ){
        if((row < width && col < height)){
            atomicAdd(&(out[histo_private[threadIdx.y][threadIdx.x]]),1);
        }
    }
}
__global__ void normalize(unsigned char * out,unsigned int * in,int size,unsigned int min,unsigned int max){
    int i = threadIdx.x +blockIdx.x * blockDim.x;
    if(i<size){
        float tmp = ((((in[i]))/float(max-min)))*255;
        unsigned char normalized= tmp;
        out[i] = normalized;
    }
}
unsigned int getMin(unsigned int * array,int aSize){
    unsigned int min=array[0];
    for(int i=1;i<aSize;i++){
        if(array[i]<min){
            min=array[i];
        }
    }
    return min;
}
unsigned int getMax(unsigned int * array,int aSize){
    unsigned int max=array[0];
    for(int i=1;i<aSize;i++){
        if(array[i]>max){
            max=array[i];
        }
    }
    return max;
}

int main(int argc, char *argv[]){


    std::cout << "**********************************"<<std::endl;
    std::cout << "*Histogram creator on \x1B[32mGPU\033[0m        *"<<std::endl;    
    std::cout << "*                                *"<<std::endl;
    std::cout << "*Author: Martin Trinacty         *"<<std::endl;
    std::cout << "**********************************"<<std::endl;
    if(argc<=1){
        std::cout << "*                                *"<<std::endl;
        std::cout << "*\x1B[31mPlease supply path to image\033[0m     *"<<std::endl;
        std::cout << "*\x1B[31mto create histogram as command\033[0m  *"<<std::endl;
        std::cout << "*\x1B[31mline argument\033[0m                   *"<<std::endl;
        std::cout << "*\x1B[31mand 1 if you want to\033[0m            *"<<std::endl;
        std::cout << "*\x1B[31muse CUDA streams\033[0m                *"<<std::endl;
        std::cout << "*                                *"<<std::endl;
        std::cout << "**********************************"<<std::endl;
        return 1;
    }

    png::image<png::rgb_pixel> img((argv[1]));

    unsigned int width= img.get_width();
    unsigned int height= img.get_height();
    
    int size = width*height*sizeof(unsigned char);
    int size_img = width_img*height_img*3*sizeof(unsigned char);
    bool stream=false;
    if(argc==3 && strcmp(argv[2],"1")==0){
        stream=true;
        std::cout << "*                                *"<<std::endl;
        std::cout << "*\x1B[32mUsing streams\033[0m                   *"<<std::endl;
        std::cout << "*                                *"<<std::endl;
        std::cout << "**********************************"<<std::endl;
    }
    
    cudaStream_t r_stream, g_stream, b_stream;
    

    unsigned char *h_r = new unsigned char [size];
    unsigned char *h_g = new unsigned char [size];
    unsigned char *h_b = new unsigned char [size];


    unsigned char *h_r_n = new unsigned char [size];
    unsigned char *h_g_n = new unsigned char [size];
    unsigned char *h_b_n = new unsigned char [size];

    unsigned int *h_histogram_r = new unsigned int [256];
    unsigned int *h_histogram_g = new unsigned int [256];
    unsigned int *h_histogram_b = new unsigned int [256];

    unsigned char *h_histogram_rN = new unsigned char [256];
    unsigned char *h_histogram_gN = new unsigned char [256];
    unsigned char *h_histogram_bN = new unsigned char [256];
    
    unsigned char * h_imgR = new unsigned char [size_img];
    unsigned char * h_imgG = new unsigned char [size_img]; 
    unsigned char * h_imgB = new unsigned char [size_img];     
    /*Allocate memory on GPU*/

    unsigned char *d_r = NULL;
    unsigned char *d_g = NULL;
    unsigned char *d_b = NULL;

    size_t pitch_r = 0;
    size_t pitch_g = 0;
    size_t pitch_b = 0;
    
    unsigned char *d_r_n = NULL;
    unsigned char *d_g_n = NULL;
    unsigned char *d_b_n = NULL;


    unsigned int *d_histogram_r = NULL;
    unsigned int *d_histogram_g = NULL;
    unsigned int *d_histogram_b = NULL;

    unsigned char * d_imgR = NULL;
    unsigned char * d_imgG = NULL;
    unsigned char * d_imgB = NULL;


    if(stream){
        CUDA_CHECK_RETURN(cudaStreamCreate(&r_stream));
        CUDA_CHECK_RETURN(cudaStreamCreate(&g_stream));
        CUDA_CHECK_RETURN(cudaStreamCreate(&b_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(r_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(g_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(b_stream));
    }
    pvg::pngToRgb3(h_r,h_g,h_b,img);
    
    CUDA_CHECK_RETURN(cudaMalloc(&d_imgR,size_img));
    CUDA_CHECK_RETURN(cudaMalloc(&d_imgG,size_img));
    CUDA_CHECK_RETURN(cudaMalloc(&d_imgB,size_img));

    CUDA_CHECK_RETURN(cudaMallocPitch(&d_r,&pitch_r,width,height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_g,&pitch_g,width,height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_b,&pitch_b,width,height));



    CUDA_CHECK_RETURN(cudaMalloc(&d_r_n,256*(sizeof(unsigned char))));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_n,256*(sizeof(unsigned char))));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_n,256*(sizeof(unsigned char))));

    CUDA_CHECK_RETURN(cudaMalloc(&d_histogram_r,size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_histogram_g,size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_histogram_b,size));
    std::cout << "*                                *"<<std::endl;
    std::cout << "*Memory allocation \x1B[32mdone\033[0m          *"<<std::endl;
    std::cout << "*                                *"<<std::endl;
    std::cout << "**********************************"<<std::endl;
    if(stream){
        CUDA_CHECK_RETURN(cudaMemcpy2DAsync(d_r,pitch_r,h_r,width,width,height,cudaMemcpyHostToDevice,r_stream));
        CUDA_CHECK_RETURN(cudaMemcpy2DAsync(d_g,pitch_g,h_g,width,width,height,cudaMemcpyHostToDevice,g_stream));
        CUDA_CHECK_RETURN(cudaMemcpy2DAsync(d_b,pitch_b,h_b,width,width,height,cudaMemcpyHostToDevice,b_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(r_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(g_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(b_stream));
    }
    else{
        CUDA_CHECK_RETURN(cudaMemcpy2D(d_r,pitch_r,h_r,width,width,height,cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy2D(d_g,pitch_g,h_g,width,width,height,cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy2D(d_b,pitch_b,h_b,width,width,height,cudaMemcpyHostToDevice));
    }
    //dim3 gridSize((width+ TILE_SIZE -1)/TILE_SIZE,(height+ TILE_SIZE -1)/TILE_SIZE);

    dim3 gridSize((width+Block_SIZE -1)/Block_SIZE,(height+Block_SIZE -1)/Block_SIZE);
    dim3 blockSize(Block_SIZE,Block_SIZE);

    histogram<<<gridSize,blockSize>>>(d_histogram_r,d_r,pitch_r,width,height);
    histogram<<<gridSize,blockSize>>>(d_histogram_g,d_g,pitch_g,width,height);
    histogram<<<gridSize,blockSize>>>(d_histogram_b,d_b,pitch_b,width,height);

    std::cout << "*                                *"<<std::endl;
    std::cout << "*Histogram computation \x1B[32mdone\033[0m      *"<<std::endl;
    std::cout << "*                                *"<<std::endl;
    std::cout << "**********************************"<<std::endl;

    /*Normalize histogram into 0 to 256*/
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    if(stream){
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_histogram_r,d_histogram_r,256*(sizeof(unsigned int)),cudaMemcpyDeviceToHost,r_stream));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_histogram_g,d_histogram_g,256*(sizeof(unsigned int)),cudaMemcpyDeviceToHost,g_stream));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_histogram_b,d_histogram_b,256*(sizeof(unsigned int)),cudaMemcpyDeviceToHost,b_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(r_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(g_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(b_stream));
    }
    else{
        CUDA_CHECK_RETURN(cudaMemcpy(h_histogram_r,d_histogram_r,256*(sizeof(unsigned int)),cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(h_histogram_g,d_histogram_g,256*(sizeof(unsigned int)),cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(h_histogram_b,d_histogram_b,256*(sizeof(unsigned int)),cudaMemcpyDeviceToHost));
    }
    unsigned int rMax= getMax(h_histogram_r,256);
    unsigned int gMax= getMax(h_histogram_g,256);
    unsigned int bMax= getMax(h_histogram_b,256);

    unsigned int rMin= getMin(h_histogram_r,256);
    unsigned int gMin= getMin(h_histogram_g,256);
    unsigned int bMin= getMin(h_histogram_b,256);
    
    int blockSizeN = Block_SIZE;
    int gridSizeN = (256+Block_SIZE-1)/Block_SIZE;
    normalize<<<gridSizeN,blockSizeN>>>(d_r_n, d_histogram_r,256, rMin,rMax);
    normalize<<<gridSizeN,blockSizeN>>>(d_g_n, d_histogram_g,256, gMin,gMax);
    normalize<<<gridSizeN,blockSizeN>>>(d_b_n, d_histogram_b,256, bMin,bMax);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    std::cout << "*                                *"<<std::endl;
    std::cout << "*Histogram normalization \x1B[32mdone\033[0m    *"<<std::endl;
    std::cout << "*                                *"<<std::endl;
    std::cout << "**********************************"<<std::endl;

    if(stream){
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_histogram_rN,d_r_n,256*(sizeof(unsigned char)),cudaMemcpyDeviceToHost,r_stream));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_histogram_gN,d_g_n,256*(sizeof(unsigned char)),cudaMemcpyDeviceToHost,g_stream));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_histogram_bN,d_b_n,256*(sizeof(unsigned char)),cudaMemcpyDeviceToHost,b_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(r_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(g_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(b_stream));
    }
    else{
        CUDA_CHECK_RETURN(cudaMemcpy(h_histogram_rN,d_r_n,256*(sizeof(unsigned char)),cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(h_histogram_gN,d_g_n,256*(sizeof(unsigned char)),cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(h_histogram_bN,d_b_n,256*(sizeof(unsigned char)),cudaMemcpyDeviceToHost));
    }
    createImage<<<gridSize,blockSize>>>(d_r_n,d_imgR,Red,width_img,height_img);
    createImage<<<gridSize,blockSize>>>(d_g_n,d_imgG,Green,width_img,height_img);
    createImage<<<gridSize,blockSize>>>(d_b_n,d_imgB,Blue,width_img,height_img);

    std::cout << "*                                *"<<std::endl;
    std::cout << "*Creating image \x1B[32mdone\033[0m             *"<<std::endl;
    std::cout << "*                                *"<<std::endl;
    std::cout << "**********************************"<<std::endl;
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    /*Copy img back to CPU*/
    if(stream){
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_imgR,d_imgR,size_img,cudaMemcpyDeviceToHost,r_stream));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_imgG,d_imgG,size_img,cudaMemcpyDeviceToHost,g_stream));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_imgB,d_imgB,size_img,cudaMemcpyDeviceToHost,b_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(r_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(g_stream));
        CUDA_CHECK_RETURN(cudaStreamSynchronize(b_stream));
    }
    else{
        CUDA_CHECK_RETURN(cudaMemcpy(h_imgR,d_imgR,size_img,cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(h_imgG,d_imgG,size_img,cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(h_imgB,d_imgB,size_img,cudaMemcpyDeviceToHost));
    }
    /*Convert data to PNG*/

    png::image<png::rgb_pixel> imgR(width_img,height_img);
    pvg::rgbToPng(imgR,h_imgR);
    imgR.write("../histRed.png");
    png::image<png::rgb_pixel> imgG(width_img,height_img);
    pvg::rgbToPng(imgG,h_imgG);
    imgG.write("../histGreen.png");
    png::image<png::rgb_pixel> imgB(width_img,height_img);
    pvg::rgbToPng(imgB,h_imgB);
    imgB.write("../histBlue.png");

    std::cout << "*                                *"<<std::endl;
    std::cout << "*All histograms                  *"<<std::endl;
    std::cout << "*written to main folder          *"<<std::endl;
    std::cout << "*                                *"<<std::endl;
    std::cout << "**********************************"<<std::endl;
    delete[] h_imgR;
    CUDA_CHECK_RETURN(cudaFree(d_imgR));
    delete[] h_imgG;
    CUDA_CHECK_RETURN(cudaFree(d_imgG));
    delete[] h_imgB;
    CUDA_CHECK_RETURN(cudaFree(d_imgB));

    CUDA_CHECK_RETURN(cudaFree(d_r));
    CUDA_CHECK_RETURN(cudaFree(d_g));
    CUDA_CHECK_RETURN(cudaFree(d_b));
    CUDA_CHECK_RETURN(cudaFree(d_r_n));
    CUDA_CHECK_RETURN(cudaFree(d_g_n));
    CUDA_CHECK_RETURN(cudaFree(d_b_n));
    CUDA_CHECK_RETURN(cudaFree(d_histogram_r));
    CUDA_CHECK_RETURN(cudaFree(d_histogram_g));
    CUDA_CHECK_RETURN(cudaFree(d_histogram_b));

    delete [] h_r;
    delete [] h_g;
    delete [] h_b;
    delete [] h_r_n;
    delete [] h_g_n;
    delete [] h_b_n;
    delete [] h_histogram_r;
    delete [] h_histogram_g;
    delete [] h_histogram_b;
    delete [] h_histogram_rN;
    delete [] h_histogram_gN;
    delete [] h_histogram_bN;


    std::cout << "*                                *"<<std::endl;
    std::cout << "*Memory deallocation \x1B[32mdone\033[0m        *"<<std::endl;
    std::cout << "*                                *"<<std::endl;
    std::cout << "**********************************"<<std::endl;
    std::cout << "*                                *"<<std::endl;
    std::cout << "*Program created histograms      *"<<std::endl;
    std::cout << "*\x1B[32mSuccesfuly\033[0m                      *"<<std::endl;
    std::cout << "**********************************"<<std::endl;
}