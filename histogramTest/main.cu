#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include "utils/pngio.h"

#include "/usr/local/cuda/include/cuda_runtime.h"

using namespace std;

#define BLOCK_SIZE (16u)
#define HISTO_HEIGHT (256u)

#define INPUT_IMG_PATH "../"
#define OUTPUT_IMG_PATH "../images/output/"

#define CUDA_CHECK_RETURN( value ){                                                                         \
    cudaError_t err = value;                                                                                \
    if(err != cudaSuccess){                                                                                 \
        fprintf(stderr, "Error %s at line %d in file %s", cudaGetErrorString(err), __LINE__, __FILE__);     \
        exit(1);                                                                                            \
    }                                                                                                       \
}

typedef enum Image { IMAGE_LENA, IMAGE_COLORS, IMAGE_HEATMAP } Image_t;
typedef enum Color { RED, GREEN, BLUE} Color_t;
typedef enum UserChoice { C_LENA, C_LENA_STREAM, C_COLORS, C_COLORS_STREAM, C_HEATMAP, C_HEATMAP_STREAM, C_UNDEFINED, C_QUIT } UserChoice_t;


void freeHostMemory(unsigned char* h_r, unsigned char* h_g, unsigned char* h_b, unsigned int* h_r_n, unsigned int* h_g_n, unsigned int* h_b_n, unsigned char* h_r_Res, unsigned char* h_g_Res, unsigned char* h_b_Res){
    delete [] h_r;
    delete [] h_g;
    delete [] h_b;
    delete [] h_r_n;
    delete [] h_g_n;
    delete [] h_b_n;
    delete [] h_r_Res;
    delete [] h_g_Res;
    delete [] h_b_Res;
}

void freeDeviceMemory(unsigned char* d_r, unsigned char* d_g, unsigned char* d_b, unsigned int* d_r_n, unsigned int* d_g_n, unsigned int* d_b_n){
    CUDA_CHECK_RETURN(cudaFree(d_r));
    CUDA_CHECK_RETURN(cudaFree(d_g));
    CUDA_CHECK_RETURN(cudaFree(d_b));

    CUDA_CHECK_RETURN(cudaFree(d_r_n));
    CUDA_CHECK_RETURN(cudaFree(d_g_n));
    CUDA_CHECK_RETURN(cudaFree(d_b_n));
}

void allocateMemoryDevice(unsigned char* d_r, unsigned char* d_g, unsigned char* d_b, unsigned char* d_r_n, unsigned char* d_g_n, unsigned char* d_b_n, size_t* pitch_r, size_t* pitch_g, size_t* pitch_b, unsigned int width, unsigned int height, unsigned int size){
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_r, pitch_r, width, height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_g, pitch_g, width, height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_b, pitch_b, width, height));

    CUDA_CHECK_RETURN(cudaMalloc(&d_r_n, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_n, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_n, size));
}

void createCudaStreams(cudaStream_t stream_r, cudaStream_t stream_g, cudaStream_t stream_b){
    CUDA_CHECK_RETURN(cudaStreamCreate( &stream_r ));
    CUDA_CHECK_RETURN(cudaStreamCreate( &stream_g ));
    CUDA_CHECK_RETURN(cudaStreamCreate( &stream_b ));
}

void synchronizeCudaStreams(cudaStream_t stream_r, cudaStream_t stream_g, cudaStream_t stream_b){
    CUDA_CHECK_RETURN(cudaStreamSynchronize( stream_r ));
    CUDA_CHECK_RETURN(cudaStreamSynchronize( stream_g ));
    CUDA_CHECK_RETURN(cudaStreamSynchronize( stream_b ));
}

void copyArraysToDevice(unsigned char* h_r, unsigned char* h_g, unsigned char* h_b, unsigned char* d_r, unsigned char* d_g, unsigned char* d_b, size_t pitch_r, size_t pitch_g, size_t pitch_b, unsigned int width, unsigned int height){ 
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_r, pitch_r, h_r, width, width, height, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_g, pitch_g, h_g, width, width, height, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_b, pitch_b, h_b, width, width, height, cudaMemcpyHostToDevice));
}

void copyArraysToDeviceAsync(unsigned char* h_r, unsigned char* h_g, unsigned char* h_b, unsigned char* d_r, unsigned char* d_g, unsigned char* d_b, size_t pitch_r, size_t pitch_g, size_t pitch_b, unsigned int width, unsigned int height, cudaStream_t stream_r, cudaStream_t stream_g, cudaStream_t stream_b){
    CUDA_CHECK_RETURN(cudaMemcpy2DAsync(d_r, pitch_r, h_r, width, width, height, cudaMemcpyHostToDevice, stream_r));
    CUDA_CHECK_RETURN(cudaMemcpy2DAsync(d_g, pitch_g, h_g, width, width, height, cudaMemcpyHostToDevice, stream_g));
    CUDA_CHECK_RETURN(cudaMemcpy2DAsync(d_b, pitch_b, h_b, width, width, height, cudaMemcpyHostToDevice, stream_b));
}

void copyArraysToHost(unsigned int* h_r_n, unsigned int* h_g_n, unsigned int* h_b_n, unsigned int* d_r_n, unsigned int* d_g_n, unsigned int* d_b_n, unsigned int size){
    CUDA_CHECK_RETURN(cudaMemcpy(h_r_n, d_r_n, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_g_n, d_g_n, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_b_n, d_b_n, size, cudaMemcpyDeviceToHost));
}

void copyArraysToHostAsync(unsigned int* h_r_n, unsigned int* h_g_n, unsigned int* h_b_n, unsigned int* d_r_n, unsigned int* d_g_n, unsigned int* d_b_n, unsigned int size, cudaStream_t stream_r, cudaStream_t stream_g, cudaStream_t stream_b){
    CUDA_CHECK_RETURN(cudaMemcpyAsync(h_r_n, d_r_n, size, cudaMemcpyDeviceToHost, stream_r));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(h_g_n, d_g_n, size, cudaMemcpyDeviceToHost, stream_g));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(h_b_n, d_b_n, size, cudaMemcpyDeviceToHost, stream_b));
}

void initializeResultArrays(unsigned char* h_r_res, unsigned char* h_g_res, unsigned char* h_b_res, unsigned int resSize){
	for (unsigned int i = 0; i < resSize; i++) {
		h_r_res[i] = h_g_res[i] = h_b_res[i] = 0;
	}
}

void clearAllocatedMem(unsigned int* h_r_n, unsigned int* h_g_n, unsigned int* h_b_n, unsigned int histoSize){
    for (unsigned int i = 0; i < HISTO_HEIGHT; i++) {
		h_r_n[i] = h_g_n[i] = h_b_n[i] = 0;
	}
}

void normalizeHistogramImageSize(unsigned int* h_r_n, unsigned int * h_g_n, unsigned int* h_b_n){
	unsigned int m_r = 0; unsigned int m_g = 0; unsigned int m_b = 0;
	
	for(unsigned int i = 0; i < HISTO_HEIGHT; i++){
        m_r = m_r < h_r_n[i] ? h_r_n[i] : m_r;
        m_g = m_g < h_g_n[i] ? h_g_n[i] : m_g;
        m_b = m_b < h_b_n[i] ? h_b_n[i] : m_b;
	}

	for(unsigned int i = 0; i < HISTO_HEIGHT; i++){
		h_r_n[i] = ((double)h_r_n[i] / m_r) * HISTO_HEIGHT;
		h_g_n[i] = ((double)h_g_n[i] / m_g) * HISTO_HEIGHT;
		h_b_n[i] = ((double)h_b_n[i] / m_b) * HISTO_HEIGHT;
    }
}

void createHistogramImgData(unsigned int* h_r_n, unsigned int* h_g_n, unsigned int* h_b_n, unsigned char* h_r_res, unsigned char* h_g_res, unsigned char* h_b_res, Color_t selectedColor){
    unsigned int cr; unsigned int cg; unsigned int cb;
    const unsigned int colorMin = 0; const unsigned int colorMax = 255;
	
	for(unsigned int i = 0; i < HISTO_HEIGHT; ++i){
		cr = h_r_n[i]; cg = h_g_n[i]; cb = h_b_n[i];
		
		for(unsigned int j = 0; j < HISTO_HEIGHT; ++j){
            unsigned int cell = HISTO_HEIGHT * ((HISTO_HEIGHT - 1) - j) + i;

            switch(selectedColor){
                case RED:
                    if(j < cr){ h_r_res[cell] = colorMax; }
                    else{ h_r_res[cell] = colorMin; }
                    h_g_res[cell] = h_b_res[cell] = colorMin;
                break;

                case GREEN:
                    if(j < cg){ h_g_res[cell] = colorMax; }
                    else{ h_g_res[cell] = colorMin; }
                    h_r_res[cell] = h_b_res[cell] = colorMin;
                break;

                case BLUE:
                    if(j < cb){ h_b_res[cell] = colorMax; }
                    else{ h_b_res[cell] = colorMin; }
                    h_r_res[cell] = h_g_res[cell] = colorMin;
                break;
            }
		}
	}	
}

string getImageOutPath(Image_t selectedImage, vector<string> imgNames){
    int selectedImgIndex = (int)selectedImage;
    string name = string(OUTPUT_IMG_PATH) + imgNames[selectedImgIndex] + "/";
    return name;
}

void createHistogramOutputs(Image_t image, vector<string> imgNames, unsigned int* h_r_n, unsigned int* h_g_n, unsigned int* h_b_n, unsigned char* h_r_res, unsigned char* h_g_res, unsigned char* h_b_res){
    vector<string> colorNames;

    colorNames.push_back("R");
    colorNames.push_back("G");
    colorNames.push_back("B");

    for(unsigned int i = 0; i < colorNames.size(); i++){
        Color_t currentColor = (Color_t)i;

        initializeResultArrays(h_r_res, h_g_res, h_b_res, HISTO_HEIGHT * HISTO_HEIGHT);

        createHistogramImgData(h_r_n, h_g_n, h_b_n, h_r_res, h_g_res, h_b_res, currentColor);    
        png::image<png::rgb_pixel> imgRes(HISTO_HEIGHT, HISTO_HEIGHT);
        pvg::rgb3ToPng(imgRes, h_r_res, h_g_res, h_b_res);

        imgRes.write(getImageOutPath(image, imgNames) + imgNames[(int)image] + "_Histogram_" + colorNames[i] + ".png");
    }
}

vector<string> initializeNames(){
    vector<string> imgNames;

    imgNames.push_back("lena");
    imgNames.push_back("Colors");
    imgNames.push_back("Heatmap");

    return imgNames;
}

void printMenu(vector<string>* opNames){
    cout << "----------------------------------------------------" << endl;
    cout << "Welcome to Histogram Creator Ultimate." << endl;
    cout << "Which breathtaking imagery would you like to analyze today?" << endl;
    cout << "----------------------------------------------------" << endl;

    for(int i = 0; i < opNames->size(); ++i){
        cout << "    " << (i*2 + 1) << ") - '" << opNames->at(i) << ".png'" << endl;
        cout << "    " << (i*2 + 2) << ") - '" << opNames->at(i) << ".png'" << " [Stream]" << endl;
    }

    cout << "    Q) - EXIT PROGRAM" << endl;
    cout << "----------------------------------------------------" << endl;
}


UserChoice_t getOperation(){
    
    char ch;
    if (cin.bad()) {
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }
    else{ cin >> ch; }

    string parsedOp;
    parsedOp.push_back(toupper(ch));
    UserChoice_t choice = C_UNDEFINED;

    if(parsedOp.compare("1") == 0){ choice = C_LENA; }
    else if(parsedOp.compare("2") == 0){ choice = C_LENA_STREAM; }
    else if(parsedOp.compare("3") == 0){ choice = C_COLORS; }
    else if(parsedOp.compare("4") == 0){ choice = C_COLORS_STREAM; }
    else if(parsedOp.compare("5") == 0){ choice = C_HEATMAP; }
    else if(parsedOp.compare("6") == 0){ choice = C_HEATMAP_STREAM; }
    else if(parsedOp.compare("Q") == 0){ choice = C_QUIT; }
    else{ choice = C_UNDEFINED; }

    return choice;
}


__global__ void processColorChannel(unsigned int* d_channel_n, unsigned char * d_channel, size_t pitch, unsigned int width, unsigned int height){

    __shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;

    if((x >= 0) && (x < width) && (y >= 0) && (y < height)){
        sBuffer[threadIdx.y][threadIdx.x] = d_channel[y * pitch + x];
    }
    else{
        sBuffer[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    if(threadIdx.x < BLOCK_SIZE && threadIdx.y < BLOCK_SIZE){
        if(x < width && y < height){
            atomicAdd(&(d_channel_n[sBuffer[threadIdx.y][threadIdx.x]]), 1);
        }
    }

    /* Solution with global memory */

	// int index = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	// if(index < width){
	// 	atomicAdd(&(d_channel_n[d_channel[index]]), 1);
	// }
}


double imageProcessingRoutine(Image_t image, vector<string> imgNames, bool stream){

    // load image, perform size calculations
    png::image<png::rgb_pixel> img("../lena.png");

    const unsigned int width = img.get_width();
    const unsigned int height = img.get_height();

    const unsigned int size = width * height * sizeof(unsigned char);
    const unsigned int histoSize = HISTO_HEIGHT * sizeof(unsigned int);

    // initialize original and new host arrays for individual color channels
    const unsigned int histoImgSize = pow(HISTO_HEIGHT, 2);
    unsigned char *h_r = new unsigned char [size]; unsigned char *h_g = new unsigned char [size]; unsigned char *h_b = new unsigned char [size];
    unsigned int *h_r_n = new unsigned int [histoSize]; unsigned int *h_g_n = new unsigned int [histoSize]; unsigned int *h_b_n = new unsigned int [histoSize];
    unsigned char *h_r_res = new unsigned char[histoImgSize]; unsigned char *h_g_res = new unsigned char[histoImgSize]; unsigned char *h_b_res = new unsigned char[histoImgSize];

    pvg::pngToRgb3(h_r, h_g, h_b, img);
    clearAllocatedMem(h_r_n, h_g_n, h_b_n, histoSize);

    // initialize original and new device arrays for individual color channels
    unsigned char *d_r = NULL; unsigned char *d_g = NULL; unsigned char *d_b = NULL;
    unsigned int *d_r_n = NULL; unsigned int *d_g_n = NULL; unsigned int *d_b_n = NULL;

    // initialize pitch variables
    size_t pitch_r, pitch_g, pitch_b;
    pitch_r = pitch_g = pitch_b = 0;

    // preparations ready, start up timer
    clock_t t = clock();

    cudaStream_t stream_r, stream_g, stream_b;
    stream_r = NULL; stream_g = NULL; stream_b = NULL;
    if(stream){
        createCudaStreams(stream_r, stream_g, stream_b);
    }

    // allocate memory, copy arrays
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_r, &pitch_r, width, height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_g, &pitch_g, width, height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_b, &pitch_b, width, height));

    CUDA_CHECK_RETURN(cudaMalloc(&d_r_n, histoSize));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_n, histoSize));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_n, histoSize));

    std::cout <<sizeof(d_r)<<std::endl <<sizeof(h_r) <<std::endl<<pitch_b<<std::endl;

    if(stream){
        copyArraysToDeviceAsync(h_r, h_g, h_b, d_r, d_g, d_b, pitch_r, pitch_g, pitch_b, width, height, stream_r, stream_g, stream_b);
    }
    else {
        copyArraysToDevice(h_r, h_g, h_b, d_r, d_g, d_b, pitch_r, pitch_g, pitch_b, width, height);
    }

    // initialize GPU kernel
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    if(stream){
        processColorChannel<<<gridSize, blockSize, 0, stream_r>>>(d_r_n, d_r, pitch_r, width, height);
        processColorChannel<<<gridSize, blockSize, 0, stream_g>>>(d_g_n, d_g, pitch_g, width, height);
        processColorChannel<<<gridSize, blockSize, 0, stream_b>>>(d_b_n, d_b, pitch_b, width, height);
    }
    else{
        processColorChannel<<<gridSize, blockSize>>>(d_r_n, d_r, pitch_r, width, height);
        processColorChannel<<<gridSize, blockSize>>>(d_g_n, d_g, pitch_g, width, height);
        processColorChannel<<<gridSize, blockSize>>>(d_b_n, d_b, pitch_b, width, height);
    }

    // run kernel
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    if(stream){
        copyArraysToHostAsync(h_r_n, h_g_n, h_b_n, d_r_n, d_g_n, d_b_n, histoSize, stream_r, stream_g, stream_b);
        synchronizeCudaStreams(stream_r, stream_g, stream_b);
    }
    else{
        copyArraysToHost(h_r_n, h_g_n, h_b_n, d_r_n, d_g_n, d_b_n, histoSize);
    }

    // main algorithm finished, only formatting work remains - stop the clock
    t = clock() - t;
    double totalTime = ((double)t) / CLOCKS_PER_SEC;

    // copy arrays back, transform to image
    normalizeHistogramImageSize(h_r_n, h_g_n, h_b_n);
    createHistogramOutputs(image, imgNames, h_r_n, h_g_n, h_b_n, h_r_res, h_g_res, h_b_res);

    // free device memory
    freeDeviceMemory(d_r, d_g, d_b, d_r_n, d_g_n, d_b_n);
    freeHostMemory(h_r, h_g, h_b, h_r_n, h_g_n, h_b_n, h_r_res, h_g_res, h_b_res);

    return totalTime;
}

int main(){
bool running = true;
    UserChoice_t selectedOperation;
    vector<string> imgNames = initializeNames();

    while(running){
        bool stream = true;
        double totalTime = 0;

        printMenu(&imgNames);
        selectedOperation = getOperation();

        switch(selectedOperation){

            case C_LENA:
                stream = false;
            case C_LENA_STREAM:
                cout << endl << "'" << imgNames[(int)IMAGE_LENA] << ".png'" << (stream? " stream" : "") << " image processing has been chosen." << endl;
                totalTime = imageProcessingRoutine(IMAGE_LENA, imgNames, stream);
                cout << "Image processing process (without load / formatting) finished in: " << totalTime << "s." << endl;
                cout << "Image histograms have been written to directory: '" << getImageOutPath(IMAGE_LENA, imgNames) << "'." << endl << endl;
            
            break;

            case C_COLORS:
                stream = false;
            case C_COLORS_STREAM:
                cout << endl << "'" << imgNames[(int)IMAGE_COLORS] << ".png'" << (stream? " stream" : "") << " image processing has been chosen." << endl;
                totalTime = imageProcessingRoutine(IMAGE_COLORS, imgNames, stream);
                cout << "Image processing process (without load / formatting) finished in: " << totalTime << "s." << endl;
                cout << "Image histograms have been written to directory: '" << getImageOutPath(IMAGE_COLORS, imgNames) << "'." << endl << endl;
            break;

            case C_HEATMAP:
                stream = false;
            case C_HEATMAP_STREAM:
                cout << endl << "'" << imgNames[(int)IMAGE_HEATMAP] << ".png'" << (stream? " stream" : "") << " image processing has been chosen." << endl;
                totalTime = imageProcessingRoutine(IMAGE_HEATMAP, imgNames, stream);
                cout << "Image processing process (without load / formatting) finished in: " << totalTime << "s." << endl;
                cout << "Image histograms have been written to directory: '" << getImageOutPath(IMAGE_HEATMAP, imgNames) << "'." << endl << endl;
            break;

            case C_QUIT:
                cout << "Exiting program ..." << endl << endl;
                running = false;
            break;

            case C_UNDEFINED:
            default:
                cout << "Selected operation is not defined. Please try again." << endl << endl;
                break;
        }
    }
    return 0;
}