#include <iostream>
#include "ConvolutionCuda.h"
using namespace std;

float checkResult(cv::Mat& m1, cv::Mat& m2){
    float sum = 0;
    for(int i = 0; i < m1.rows; i++){
        for(int j = 0; j < m1.cols; j++){
            sum = sum + m1.at<float>(i,j) - m2.at<float>(i,j);
//            if(m1.at<float>(i,j) - m2.at<float>(i,j) != 0){
//                std::cout << "position missing (i,j) (" << i << ", " << j << ") " << std::endl;
//                sum += 1;
//            }
        }
    }
    return sum;
}
float checkResult3D(cv::Mat& m1, cv::Mat& m2){
    float sum = 0;
    cv::Mat channel1[3], channel2[3];
    cv::split(m1, channel1);
    cv::split(m2, channel2);
    for(int i = 0; i < m1.rows; i++){
        for(int j = 0; j < m1.cols; j++){
            sum = sum + channel1[0].at<float>(i,j) - channel2[0].at<float>(i,j) + channel1[1].at<float>(i,j) - channel2[1].at<float>(i,j) + channel1[2].at<float>(i,j) - channel2[2].at<float>(i,j);
//            if(m1.at<float>(i,j) - m2.at<float>(i,j) != 0){
//                std::cout << "position missing (i,j) (" << i << ", " << j << ") " << std::endl;
//                sum += 1;
//            }
        }
    }
    return sum;
}

int main(int argc, char **argv)
{
    cv::Mat img = cv::imread("/home/nvidia/imgs/images_filter/972_75.791195_1608895267265.png", cv::IMREAD_COLOR);
    cv::Mat img1 = cv::imread("/home/nvidia/workspace/cuda-convolution/640x480-afframont_ujamondrone_leitosa.jpeg", cv::IMREAD_COLOR);
    std::cout << "img1.rows " << img1.rows << " img1.cols " << img1.cols << " img1.channels " << img1.channels() << std::endl;
    cv::Mat A (img, cv::Rect(35, 60, 640, 480) );
    cv::Mat B, image_filter;
    int mask_width = atoi(argv[1]);
    img1.convertTo(B, CV_32FC3);
    cv::Mat kernelH(1, mask_width, CV_32F);
    for(int i = 0; i < mask_width; i++){
        kernelH.at<float>(0,i) = i - mask_width/2;
    }
    auto start1 = getMoment;
    cv::filter2D(B, image_filter, -1, kernelH, cv::Point(-1,-1), 0.0, cv::BORDER_REPLICATE);
    auto end1 = getMoment;
//    cv::cuda::GpuMat src, dst, kernel;
//    src.upload(B);
//    kernel.upload(kernelH);

//    Ptr<cuda::Convolution> test = cuda::createConvolution();
//    auto start1 = getMoment;
//    test->convolve(src, kernel, dst);
////    cv::filter2D(B, image_filter, -1, kernelH, cv::Point(-1,-1), 0.0, cv::BORDER_REPLICATE);
//    auto end1 = getMoment;
    cout << "Convolution Time = "<< chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count()  << endl;
    ConvolutionCuda filter;
    filter.setInput(img1, kernelH);
    filter.cudaInit();
    filter.setupData();

    start1 = getMoment;
    for(int i = 0; i < 100; i++){

        filter.filterCuda();
    }
    end1 = getMoment;
    cout << "GPU Time = "<< chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count()/100  << endl;
    cv::Mat C;
    filter.getResult(C);
    std::cout << "checkResult " << checkResult3D(image_filter, C) << std::endl;
    filter.cudaFreeMem();
    cv::imshow("CPU", image_filter);
    cv::imshow("GPU", C);
//    cv::Mat channel[3];
//    cv::split(C, channel);

//    cv::imshow("GPU1", channel[0]);
//    cv::imshow("GPU2", channel[1]);
//    cv::imshow("GPU3", channel[2]);
//    cv::Mat channel1[3];
//    cv::split(image_filter, channel1);

//    cv::imshow("CPU1", channel1[0]);
//    cv::imshow("CPU2", channel1[1]);
//    cv::imshow("CPU3", channel1[2]);
    cv::waitKey(0);
    return 0;
}

