#ifndef CONVOLUTIONCUDA_H
#define CONVOLUTIONCUDA_H
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"
#include <fstream>
#include<opencv2/opencv.hpp>
#include <chrono>

#define WIDTH 640
#define HEIGHT 480
#define OVERLAP_WIDTH 80
#define getMoment std::chrono::high_resolution_clock::now()
using namespace std;
using namespace cv;


#define gpuErrChk(call) {gpuError((call));}
inline void gpuError(cudaError_t call){
    const cudaError_t error= call;
    if(error != cudaSuccess){
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}
inline void writeToFile(char * filename,int *data, int width, int height)
{
    printf("Writing to file ...\n");
    std::cout << filename << std::endl;
    FILE* file;
    file = fopen(filename, "w");
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            fprintf(file, "%d ", data[y * width + x]);
        }
        fprintf(file, "\n");
    }
//    fprintf(file, "%d ", data[width*height]);
//    fprintf(file, "%d ", data[width*height+1]);
    fclose(file);
}

struct soa{
    float *blueChannel;
    float *greenChannel;
    float *redChannel;
    int size;
};

class ConvolutionCuda
{
public:
    ConvolutionCuda();
    ConvolutionCuda(int rows, int cols);

public:
    void setInput(cv::Mat& source, cv::Mat& filter_kernel1);
    void h_mem_init();
    void d_mem_init();

    void cudaInit();
    void cudaFreeMem();

    void setupData();
    // This function compute convolution
    void filterCuda();
    void getResult(cv::Mat& dst);

public:
    cv::Mat src, filter_kernel;
    int image_width, image_height, image_padding_width, image_padding_height;
    int filter_width, filter_height, filter_radius_horizon, filter_radius_vertical;
    int image_size, size_float, padding_width;
    float *d_filter_kernel, *d_src_padding, *d_dst;
    float *h_filter_kernel, *h_src_padding, *h_dst;
    soa d_soa_src_padding, d_soa_dst, h_soa_dst, h_soa_src_padding;

};

#endif // CONVOLUTIONCUDA_H
