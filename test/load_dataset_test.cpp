#include <iostream>
#include <image_processing/BabblingDataset.h>
#include <pcl/visualization/cloud_viewer.h>
#include <string>

using namespace image_processing;

int main(int argc, char** argv){

    if(argc < 3){
        std::cerr << "usage one folder and a number of iteration" << std::endl;
        return 1;
    }

    int iteration = std::atoi(argv[2]);
    BabblingDataset bds(argv[1]);
    bds.load_dataset(iteration);

    bool clouds_end = true;

    BabblingDataset::per_iter_rgbd_set_t images = bds.get_per_iter_rgbd_set();
    BabblingDataset::rect_trajectories_set_t rects = bds.get_per_iter_rect_set();
//    BabblingDataset::cloud_trajectories_set_t cloud_data;
//    bds.extract_cloud_trajectories(cloud_data);

//    auto itr = cloud_data[iteration].begin();
    auto itr = images[iteration].begin();
    pcl::visualization::CloudViewer viewer("cloud");
    PointCloudT::Ptr cloud(new PointCloudT);
    viewer.runOnVisualizationThread([&](pcl::visualization::PCLVisualizer& vis){
        vis.removeAllPointClouds();
        cloud.reset(new PointCloudT);
        bds.rgbd_to_pointcloud(itr->second.first,itr->second.second,cloud);
//        for(int i = 0; i < itr->second.size(); i++ ){
//            cloud.reset(new PointCloudT(itr->second[i]));

//            vis.addPointCloud(cloud,"cloud_" + std::to_string(i));
//        }
        vis.addPointCloud(cloud);
    });
    while(!viewer.wasStopped() &&  clouds_end){
        cv::Mat coloured;

//        images[iteration][itr->first].first.copyTo(coloured);
        itr->second.first.copyTo(coloured);

        for(int i = 0; i < rects[iteration][itr->first].size();i++)
            cv::rectangle(coloured,rects[iteration][itr->first][i],cv::Scalar(0,0,255));
        cv::imshow("rgb",coloured);
//        cv::imshow("depth",itr_im->second.second);
        cv::waitKey(30);
//        std::cout << "type enter" << std::endl;
//        std::cin.ignore();
//        std::cout << '\r';
        itr++;
        clouds_end = itr != images[iteration].end();
    }


    return 0;

}
