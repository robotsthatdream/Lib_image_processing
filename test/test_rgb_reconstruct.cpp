#include <iostream>
#include <opencv2/opencv.hpp>
#include <image_processing/BabblingDataset.h>

using namespace image_processing;

int main(int argc, char** argv){

    if(argc < 4){
        std::cerr << "usage one folder, one dataset description yml file and a number of iteration" << std::endl;
        return 1;
    }

    int iteration = std::atoi(argv[3]);
    BabblingDataset bds(argv[1],argv[2]);
    bds.load_dataset(iteration);

    BabblingDataset::per_iter_rgbd_set_t images = bds.get_per_iter_rgbd_set();

    auto itr = images[iteration].begin();
    cv::Mat rgb;
    itr->second.first.copyTo(rgb);
    cv::Mat rgb_reconst(rgb.rows,rgb.cols,CV_8UC3);


    int cn = rgb.channels();
    for(int i = 0; i < rgb.rows; i++){
        uint8_t* rgbPtr = (uint8_t*) rgb.row(i).data;
        uint8_t* rgb_reconstPtr = (uint8_t*) rgb_reconst.row(i).data;
        for(int j = 0; j < rgb.cols; j++){
            rgb_reconstPtr[j*cn + 0] = rgbPtr[j*cn + 0];
            rgb_reconstPtr[j*cn + 1] = rgbPtr[j*cn + 1];
            rgb_reconstPtr[j*cn + 2] = rgbPtr[j*cn + 2];
        }
    }

    cv::imshow("rgb",rgb);
    cv::imshow("reconstruct",rgb_reconst);
    cv::waitKey();

    return 0;
}
