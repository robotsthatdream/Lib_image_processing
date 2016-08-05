#include <iostream>
#include <BabblingDataset.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace image_processing;

int main(int argc, char** argv){

    if(argc < 3){
        std::cerr << "usage one folder, one dataset description yml file and a number of iteration" << std::endl;
        return 1;
    }

    int iteration = std::atoi(argv[3]);
    BabblingDataset bds(argv[1],argv[2]);
    bds.load_dataset(iteration);

    BabblingDataset::per_iter_rgbd_set_t images = bds.get_per_iter_rgbd_set();
    bds._load_camera_param(std::string(argv[1]) + "/camera_parameter.yml");

    pcl::visualization::CloudViewer viewer("cloud");
    for(auto itr = images[iteration].begin(); itr != images[iteration].end(); ++itr){
        PointCloudT::Ptr cloud(new PointCloudT);
        bds._rgbd_to_pointcloud(itr->second.first,itr->second.second,cloud);
//        std::cout << cloud->size() << std::endl;
        viewer.showCloud(cloud);
        while(!viewer.wasStopped());
//        cv::imshow("rgb",itr->second.first);
//        cv::imshow("depth",itr->second.second);
//        cv::waitKey(30);
    }

    return 0;

}
