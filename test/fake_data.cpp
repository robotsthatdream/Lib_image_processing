#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>


int main(int argc,char** argv){

    if(argc < 2){
        std::cout << "bad number of arguments, see the code source" << std::endl;
        return 1;
    }

    std::cout << "CAUTION : this code will erase all the images in this folder. And replace it by 640x480 generated images" << std::endl;
    std::cout << "press enter to continue (or ctrl+c to abort)";
    std::cin.ignore();

    std::srand(std::time(NULL));

//    cv::Mat red(480,640,CV_8UC3,cv::Scalar(0,0,255));
//    cv::Mat green(480,640,CV_8UC3,cv::Scalar(0,255,0));
//    cv::Mat red_green(480,640,CV_8UC3);

//    for(int i = 0; i < red_green.cols; i+=7){
//        if(i/2*2 == i){
//            for(int j = i; j < i + 7 && j < red_green.cols; j++)
//                red.col(j).copyTo(red_green.col(j));
//        }
//        else{
//            for(int j = i; j < i + 7 && j < red_green.cols; j++)
//                green.col(j).copyTo(red_green.col(j));
//        }
//    }

    cv::Mat img(480,640,CV_8UC3);

    int size = 50;
//    for(int i = 0; i < img.rows; i+=size){
        for(int j = 0; j < img.cols; j+=size){
            int b = rand()%255, g = rand()%255, r = rand()%255;
//            for(int k = i; k < i + size &&  k < img.rows; k++)
                for(int l = j; l < j + size && l < img.cols; l++)
                    cv::Mat(480,1,CV_8UC3,cv::Scalar(b,g,r)).copyTo(img/*.row(k)*/.col(l));
        }
//    }

    boost::filesystem::directory_iterator end_itr;
    boost::filesystem::directory_iterator itr(argv[1]);
    for(; itr != end_itr; ++itr){
        cv::imwrite(itr->path().string(),img);
    }

}
