#include "ConvolutionCuda.h"
__constant__ float my_filter[7];
//ConvolutionCuda::ConvolutionCuda(int rows, int cols)
//    : height(rows), width(cols){
//    image_size = width*height;
//    size_float = sizeof(float)*image_size;


//}

//ConvolutionCuda::ConvolutionCuda(){

//}

void ConvolutionCuda::setInput(cv::Mat &source, cv::Mat &filter_kernel1){
    src = source;
    filter_kernel = filter_kernel1;
    image_width = src.cols;
    image_height = src.rows;
    filter_width = filter_kernel.cols;
    filter_height = filter_kernel.rows;
    filter_radius_horizon = filter_width/2;
    filter_radius_vertical = filter_height/2;
    if((filter_radius_horizon*2) % 4 !=0){
        padding_width = (filter_radius_horizon*2 /4 + 1) * 4;
        //padding_width = 32;
    }
    else{
        padding_width = filter_radius_horizon*2;
    }
    image_padding_width = image_width + padding_width;
    h_soa_src_padding.size = image_padding_width*image_height;
    d_soa_src_padding.size = image_padding_width*image_height;
    h_soa_dst.size = image_width*image_height;
    d_soa_dst.size = image_width*image_height;

    std::cout << "padding_width " << padding_width << " " << "image_padding_width " << image_padding_width << std::endl;


}

void ConvolutionCuda::h_mem_init()
{




    h_soa_dst.blueChannel = (float*)malloc(sizeof(float)*h_soa_dst.size);
    h_soa_dst.greenChannel = (float*)malloc(sizeof(float)*h_soa_dst.size);
    h_soa_dst.redChannel = (float*)malloc(sizeof(float)*h_soa_dst.size);

    h_soa_src_padding.blueChannel = (float*)malloc(sizeof(float)*h_soa_src_padding.size);
    h_soa_src_padding.greenChannel = (float*)malloc(sizeof(float)*h_soa_src_padding.size);
    h_soa_src_padding.redChannel = (float*)malloc(sizeof(float)*h_soa_src_padding.size);

    h_src_padding = (float*)malloc(sizeof(float)*image_padding_width*image_height);
    h_dst = (float*)malloc(sizeof(float)*image_width*image_height);
    h_filter_kernel = (float*)malloc(sizeof(float)*filter_width*filter_height);

}

void ConvolutionCuda::d_mem_init()
{
//    cudaMallocManaged((void**)&d_left_weight, size_int);
//    cudaMallocManaged((void**)&d_right_weight, size_int);
//    cudaMallocManaged((void**)&d_down_weight, size_int);
//    cudaMallocManaged((void**)&d_up_weight, size_int);




    gpuErrChk(cudaMalloc((void**)&d_soa_dst.blueChannel, sizeof(float)*d_soa_dst.size));
    gpuErrChk(cudaMalloc((void**)&d_soa_dst.greenChannel, sizeof(float)*d_soa_dst.size));
    gpuErrChk(cudaMalloc((void**)&d_soa_dst.redChannel, sizeof(float)*d_soa_dst.size));

    gpuErrChk(cudaMalloc((void**)&d_soa_src_padding.blueChannel, sizeof(float)*d_soa_src_padding.size));
    gpuErrChk(cudaMalloc((void**)&d_soa_src_padding.greenChannel, sizeof(float)*d_soa_src_padding.size));
    gpuErrChk(cudaMalloc((void**)&d_soa_src_padding.redChannel, sizeof(float)*d_soa_src_padding.size));

    gpuErrChk(cudaMalloc((void**)&d_src_padding, sizeof(float)*image_padding_width*image_height));
    gpuErrChk(cudaMalloc((void**)&d_dst, sizeof(float)*image_width*image_height));
    gpuErrChk(cudaMalloc((void**)&d_filter_kernel, sizeof(float)*filter_width*filter_height));
}

void ConvolutionCuda::cudaInit(){
    h_mem_init();
    d_mem_init();
}

void ConvolutionCuda::setupData(){
    cv::Mat m1, m2;

    (float *) m1.data ;

    src.convertTo(m1, CV_32FC3);
    cv::Mat channel1[3], channel2[3];
    cv::split(m1, channel1);
    cv::copyMakeBorder(channel1[0], channel2[0], 0, 0, padding_width-filter_radius_horizon, filter_radius_horizon, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(channel1[1], channel2[1], 0, 0, padding_width-filter_radius_horizon, filter_radius_horizon, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(channel1[2], channel2[2], 0, 0, padding_width-filter_radius_horizon, filter_radius_horizon, cv::BORDER_REPLICATE);
    //cv::imshow("copyMakeBorder", m2);
    //cv::waitKey(0);
    //m1.convertTo(m2, CV_32F);
//    std::cout << "m2.rows " << m2.rows << " m2.cols " << m2.cols << std::endl;
//    memcpy(h_src_padding, m2.ptr(0), sizeof(float)*image_padding_width*image_height);
    memcpy(h_filter_kernel, filter_kernel.ptr(0), sizeof(float)*filter_width*filter_height);
    memcpy(h_soa_src_padding.blueChannel, channel2[0].ptr(0), sizeof(float)*h_soa_src_padding.size);
    memcpy(h_soa_src_padding.greenChannel, channel2[1].ptr(0), sizeof(float)*h_soa_src_padding.size);
    memcpy(h_soa_src_padding.redChannel, channel2[2].ptr(0), sizeof(float)*h_soa_src_padding.size);

    gpuErrChk(cudaMemcpy(d_soa_src_padding.blueChannel, h_soa_src_padding.blueChannel, sizeof(float)*image_padding_width*image_height, cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_soa_src_padding.greenChannel, h_soa_src_padding.greenChannel, sizeof(float)*image_padding_width*image_height, cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_soa_src_padding.redChannel, h_soa_src_padding.redChannel, sizeof(float)*image_padding_width*image_height, cudaMemcpyHostToDevice));


    //gpuErrChk(cudaMemcpy(d_src_padding, h_src_padding, sizeof(float)*image_padding_width*image_height, cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_filter_kernel, h_filter_kernel, sizeof(float)*filter_width*filter_height , cudaMemcpyHostToDevice));
//    const float h_coef[];
//    for(int i = 0; i < filter_width; i++){
//        h_coef[i] = h_filter_kernel[i];
//    }
    cudaMemcpyToSymbol(my_filter, h_filter_kernel, filter_width * sizeof(float));
}
__global__ void
imageFilter_kernel3D(soa d_soa_src_padding, soa d_soa_dst,const float* __restrict__ d_filter_kernel, int blockDimx,
                   int filter_radius_horizon, int padding_width, int image_padding_width, int image_width){

    int ix = threadIdx.x + blockIdx.x*blockDimx;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int tid = iy*image_padding_width + ix;
//    int tid = iy*image_padding_width + ix + padding_width - filter_radius_horizon;
    __shared__ float smem[40*8];
    unsigned int smemPost = threadIdx.y*blockDim.x + threadIdx.x;
    //int idx = threadIdx.y*32 + threadIdx.x;

    smem[smemPost] = d_soa_src_padding.blueChannel[tid];
    __syncthreads();

     if(smemPost < 256){
        int x1 = smemPost & 31;
        int y1 = smemPost >> 5;
        int index = (y1 + blockIdx.y*blockDim.y)*image_width + (x1 + blockIdx.x*blockDimx);
        int smemPost1 = y1*40 + x1 + padding_width - filter_radius_horizon;
        float result = 0;
        result = smem[smemPost1 - 3]*d_filter_kernel[filter_radius_horizon -3] + \
                 smem[smemPost1 - 2]*d_filter_kernel[filter_radius_horizon -2] + \
                 smem[smemPost1 - 1]*d_filter_kernel[filter_radius_horizon -1] + \
                 smem[smemPost1 ]*d_filter_kernel[filter_radius_horizon ] + \
                 smem[smemPost1 + 1]*d_filter_kernel[filter_radius_horizon +1] + \
                 smem[smemPost1 + 2]*d_filter_kernel[filter_radius_horizon + 2] + \
                 smem[smemPost1 + 3]*d_filter_kernel[filter_radius_horizon + 3];
//        result = smem[smemPost1 - 3]*my_filter[filter_radius_horizon -3] + \
//                 smem[smemPost1 - 2]*my_filter[filter_radius_horizon -2] + \
//                 smem[smemPost1 - 1]*my_filter[filter_radius_horizon -1] + \
//                 smem[smemPost1 ]*my_filter[filter_radius_horizon ] + \
//                 smem[smemPost1 + 1]*my_filter[filter_radius_horizon +1] + \
//                 smem[smemPost1 + 2]*my_filter[filter_radius_horizon + 2] + \
//                 smem[smemPost1 + 3]*my_filter[filter_radius_horizon + 3];
        d_soa_dst.blueChannel[index] = result;
     }
     __syncthreads();

     smem[smemPost] = d_soa_src_padding.greenChannel[tid];
     __syncthreads();
     if(smemPost < 256){
        int x1 = smemPost & 31;
        int y1 = smemPost >> 5;
        int index = (y1 + blockIdx.y*blockDim.y)*image_width + (x1 + blockIdx.x*blockDimx);
        int smemPost1 = y1*40 + x1 + padding_width - filter_radius_horizon;
        float result = 0;
        result = smem[smemPost1 - 3]*d_filter_kernel[filter_radius_horizon -3] + \
                 smem[smemPost1 - 2]*d_filter_kernel[filter_radius_horizon -2] + \
                 smem[smemPost1 - 1]*d_filter_kernel[filter_radius_horizon -1] + \
                 smem[smemPost1 ]*d_filter_kernel[filter_radius_horizon ] + \
                 smem[smemPost1 + 1]*d_filter_kernel[filter_radius_horizon +1] + \
                 smem[smemPost1 + 2]*d_filter_kernel[filter_radius_horizon + 2] + \
                 smem[smemPost1 + 3]*d_filter_kernel[filter_radius_horizon + 3];
//        result = smem[smemPost1 - 3]*my_filter[filter_radius_horizon -3] + \
//                 smem[smemPost1 - 2]*my_filter[filter_radius_horizon -2] + \
//                 smem[smemPost1 - 1]*my_filter[filter_radius_horizon -1] + \
//                 smem[smemPost1 ]*my_filter[filter_radius_horizon ] + \
//                 smem[smemPost1 + 1]*my_filter[filter_radius_horizon +1] + \
//                 smem[smemPost1 + 2]*my_filter[filter_radius_horizon + 2] + \
//                 smem[smemPost1 + 3]*my_filter[filter_radius_horizon + 3];
        d_soa_dst.greenChannel[index] = result;
     }
     __syncthreads();

     smem[smemPost] = d_soa_src_padding.redChannel[tid];
     __syncthreads();
     if(smemPost < 256){
        int x1 = smemPost & 31;
        int y1 = smemPost >> 5;
        int index = (y1 + blockIdx.y*blockDim.y)*image_width + (x1 + blockIdx.x*blockDimx);
        int smemPost1 = y1*40 + x1 + padding_width - filter_radius_horizon;
        float result = 0;
        result = smem[smemPost1 - 3]*d_filter_kernel[filter_radius_horizon -3] + \
                 smem[smemPost1 - 2]*d_filter_kernel[filter_radius_horizon -2] + \
                 smem[smemPost1 - 1]*d_filter_kernel[filter_radius_horizon -1] + \
                 smem[smemPost1 ]*d_filter_kernel[filter_radius_horizon ] + \
                 smem[smemPost1 + 1]*d_filter_kernel[filter_radius_horizon +1] + \
                 smem[smemPost1 + 2]*d_filter_kernel[filter_radius_horizon + 2] + \
                 smem[smemPost1 + 3]*d_filter_kernel[filter_radius_horizon + 3];
//        result = smem[smemPost1 - 3]*my_filter[filter_radius_horizon -3] + \
//                 smem[smemPost1 - 2]*my_filter[filter_radius_horizon -2] + \
//                 smem[smemPost1 - 1]*my_filter[filter_radius_horizon -1] + \
//                 smem[smemPost1 ]*my_filter[filter_radius_horizon ] + \
//                 smem[smemPost1 + 1]*my_filter[filter_radius_horizon +1] + \
//                 smem[smemPost1 + 2]*my_filter[filter_radius_horizon + 2] + \
//                 smem[smemPost1 + 3]*my_filter[filter_radius_horizon + 3];
        d_soa_dst.redChannel[index] = result;
     }

}

//__global__ void
//imageFilter_kernel(float* d_src_padding, float* d_dst, float* d_filter_kernel, int blockDimx,
//                   int filter_radius_horizon, int padding_width, int image_padding_width, int image_width){

//    int ix = threadIdx.x + blockIdx.x*blockDimx;
//    int iy = threadIdx.y + blockIdx.y*blockDim.y;
//    int tid = iy*image_padding_width + ix;
////    int tid = iy*image_padding_width + ix + padding_width - filter_radius_horizon;
////     float result = 0;
////     result = d_src_padding[tid - 3]*my_filter[filter_radius_horizon + 3] + \
////             d_src_padding[tid - 2]*my_filter[filter_radius_horizon + 2] + \
////             d_src_padding[tid - 1]*my_filter[filter_radius_horizon + 1] + \
////             d_src_padding[tid ]*my_filter[filter_radius_horizon ] + \
////             d_src_padding[tid + 1]*my_filter[filter_radius_horizon - 1] + \
////             d_src_padding[tid + 2]*my_filter[filter_radius_horizon - 2] + \
////             d_src_padding[tid + 3]*my_filter[filter_radius_horizon - 3];
////     d_dst[iy*image_width + ix] = result;
//    __shared__ float smem[40*8];
//    unsigned int smemPost = threadIdx.y*blockDim.x + threadIdx.x;
//    //int idx = threadIdx.y*32 + threadIdx.x;
//    smem[smemPost] = d_src_padding[tid];
//    __syncthreads();

////    __shared__ float filter[7];

////    if(threadIdx.y == 0 && threadIdx.x >=0 && threadIdx.x < 7){
////        filter[threadIdx.x] = d_filter_kernel[threadIdx.x];
////    }
////    __syncthreads();

////     if(threadIdx.x >= padding_width - filter_radius_horizon && threadIdx.x < blockDim.x - filter_radius_horizon){
////         float result = 0;
////         result = smem[smemPost - 3]*my_filter[filter_radius_horizon - 3] + \
////                 smem[smemPost - 2]*my_filter[filter_radius_horizon - 2] + \
////                 smem[smemPost - 1]*my_filter[filter_radius_horizon - 1] + \
////                 smem[smemPost ]*my_filter[filter_radius_horizon ] + \
////                 smem[smemPost + 1]*my_filter[filter_radius_horizon + 1] + \
////                 smem[smemPost + 2]*my_filter[filter_radius_horizon + 2] + \
////                 smem[smemPost + 3]*my_filter[filter_radius_horizon + 3];

//////         for(int i = -filter_radius_horizon; i <= filter_radius_horizon; i++){
//////             //result += smem[smemPost + i]*d_filter_kernel[filter_radius_horizon - i];
//////             result += smem[smemPost + i]*2;
//////         }

////         d_dst[iy*image_width + ix - padding_width + filter_radius_horizon] = result;
////     }
//     if(smemPost < 256){
//        int x1 = smemPost & 31;
//        int y1 = smemPost >> 5;
//        int index = (y1 + blockIdx.y*blockDim.y)*image_width + (x1 + blockIdx.x*blockDimx);
//        int smemPost1 = y1*40 + x1 + padding_width - filter_radius_horizon;
//        float result = 0;
//        result = smem[smemPost1 - 3]*my_filter[filter_radius_horizon -3] + \
//                 smem[smemPost1 - 2]*my_filter[filter_radius_horizon -2] + \
//                 smem[smemPost1 - 1]*my_filter[filter_radius_horizon -1] + \
//                 smem[smemPost1 ]*my_filter[filter_radius_horizon ] + \
//                 smem[smemPost1 + 1]*my_filter[filter_radius_horizon +1] + \
//                 smem[smemPost1 + 2]*my_filter[filter_radius_horizon + 2] + \
//                 smem[smemPost1 + 3]*my_filter[filter_radius_horizon + 3];
////        result = smem[smemPost1 - 3]*my_filter[filter_radius_horizon -3];
//        d_dst[index] = result;
//     }

//}

void ConvolutionCuda::filterCuda(){

    //setupData(src, filter_kernel);
    int blockDimx = 32;
    int blockDimPadding = 32 + padding_width;
    dim3 block(blockDimPadding, 8, 1);
    dim3 grid((image_width+blockDimx-1)/blockDimx, (image_height+block.y-1)/block.y, 1);

//    imageFilter_kernel<<<grid, block>>>(d_src_padding, d_dst, d_filter_kernel, blockDimx,
//                                        filter_radius_horizon, padding_width, image_padding_width, image_width);
    imageFilter_kernel3D<<<grid, block>>>(d_soa_src_padding, d_soa_dst, d_filter_kernel, blockDimx,
                                        filter_radius_horizon, padding_width, image_padding_width, image_width);
    gpuErrChk(cudaDeviceSynchronize());

//    cv::Mat result1;
//    result.convertTo(result1, CV_8UC1);

//    cv::imshow("result", result);
//    cv::waitKey(0);

}

void ConvolutionCuda::getResult(Mat &dst){
//    gpuErrChk(cudaMemcpy(h_dst, d_dst, sizeof(float)*image_width*image_height , cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(h_soa_dst.blueChannel, d_soa_dst.blueChannel, sizeof(float)*image_width*image_height , cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(h_soa_dst.greenChannel, d_soa_dst.greenChannel, sizeof(float)*image_width*image_height , cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(h_soa_dst.redChannel, d_soa_dst.redChannel, sizeof(float)*image_width*image_height , cudaMemcpyDeviceToHost));
    gpuErrChk(cudaDeviceSynchronize());
    cv::Mat channel[3];
    channel[0] = cv::Mat(image_height, image_width,CV_32F, h_soa_dst.blueChannel);
    channel[1] = cv::Mat(image_height, image_width,CV_32F, h_soa_dst.greenChannel);
    channel[2] = cv::Mat(image_height, image_width,CV_32F, h_soa_dst.redChannel);
    cv::Mat result;
    cv::merge(channel, 3, result);
//    cv::Mat result(dst.rows, dst.cols, CV_32FC1, h_dst);
    result.copyTo(dst);
}

void ConvolutionCuda::cudaFreeMem()
{

    free(h_filter_kernel);
    free(h_src_padding);
    free(h_dst);
    free(h_soa_src_padding.blueChannel);
    free(h_soa_src_padding.greenChannel);
    free(h_soa_src_padding.redChannel);
    free(h_soa_dst.blueChannel);
    free(h_soa_dst.greenChannel);
    free(h_soa_dst.redChannel);

    cudaFree(d_filter_kernel);
    cudaFree(d_src_padding);
    cudaFree(d_dst);

    cudaFree(d_soa_src_padding.blueChannel);
    cudaFree(d_soa_src_padding.greenChannel);
    cudaFree(d_soa_src_padding.redChannel);
    cudaFree(d_soa_dst.blueChannel);
    cudaFree(d_soa_dst.greenChannel);
    cudaFree(d_soa_dst.redChannel);

}
