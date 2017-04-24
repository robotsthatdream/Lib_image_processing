#include <iostream>
#include <pcl/io/pcd_io.h>
#include <image_processing/HistogramFactory.hpp>
#include <image_processing/SurfaceOfInterest.h>

using namespace  image_processing;

int main(int argc, char** argv){

    if(argc < 3){
        std::cout << "usage one path for pcd file and a nbr of bins" << std::endl;
        return 1;
    }

    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    reader.read(argv[1], *cloud);

    SurfaceOfInterest soi;
    soi.setInputCloud(cloud);
    soi.computeSupervoxel();

    Eigen::MatrixXd bounds(2,3);
    bounds << 0,0,0,
              1,1,1;
    HistogramFactory hf(std::stoi(argv[2]),3,bounds);

    for(auto it = soi.getSupervoxels().begin(); it != soi.getSupervoxels().end(); ++it){
        hf.compute(it->second);
        HistogramFactory::_histogram_t histo1 = hf.get_histogram();
        std::cout << "color histograms on rgb with " << argv[2] << " bins" << std::endl;
        std::cout << "Hue : " << histo1[0].transpose() << std::endl;
        std::cout << "______" << std::endl;
        std::cout << "Saturation : " << histo1[1].transpose() << std::endl;
        std::cout << "______" << std::endl;
        std::cout << "Value : " << histo1[2].transpose() << std::endl;
        std::cout << "______" << std::endl;
    }

    return 0;
}
