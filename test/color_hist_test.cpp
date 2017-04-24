#include <iostream>
#include <opencv2/opencv.hpp>
#include <image_processing/HistogramFactory.hpp>
#include <eigen3/Eigen/Eigen>

using namespace image_processing;

int main(int argc, char** argv){

    if(argc < 3){
        std::cerr << "usage : <path_to_image> <nbr_bins> (optional <path_to_image>" << std::endl;
        return 1;
    }

    cv::Mat image, image2;
    image = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);

    if(! image.data )
    {
        std::cerr <<  "Could not open or find the image" << std::endl ;
        return 2;
    }

    Eigen::MatrixXd bounds(2,3);
    bounds << 0  ,0  ,0  ,
              255,255,255;
    HistogramFactory hf(std::stoi(argv[2]),3,bounds);
    hf.compute(image);
    HistogramFactory::_histogram_t histo1 = hf.get_histogram();

    std::cout << "color histograms on rgb with " << argv[2] << " bins" << std::endl;
    std::cout << "RED : " << histo1[0].transpose() << std::endl;
    std::cout << "______" << std::endl;
    std::cout << "GREEN : " << histo1[1].transpose() << std::endl;
    std::cout << "______" << std::endl;
    std::cout << "BLUE : " << histo1[2].transpose() << std::endl;
    std::cout << "______" << std::endl;


    if(argc == 4){

        image2 = cv::imread(argv[3],CV_LOAD_IMAGE_COLOR);

        if(! image2.data )
        {
            std::cerr <<  "Could not open or find the image2" << std::endl ;
            return 3;
        }


        hf.compute(image2);
        std::cout << "image 2 : color histograms on rgb with " << argv[2] << " bins" << std::endl;
        std::cout << "RED : " << hf.get_histogram()[0].transpose() << std::endl;
        std::cout << "______" << std::endl;
        std::cout << "GREEN : " << hf.get_histogram()[1].transpose() << std::endl;
        std::cout << "______" << std::endl;
        std::cout << "BLUE : " << hf.get_histogram()[2].transpose() << std::endl;
        std::cout << "______" << std::endl;

        std::cout << "distance between the both images" << std::endl;

        double dist[3];
        for(int i = 0; i < 3; i++){
            dist[i] = HistogramFactory::chi_squared_distance(hf.get_histogram()[i],histo1[i]);
            std::cout << dist[i] << " ; ";
        }
        std::cout << std::endl;

    }

    return 0;
}
