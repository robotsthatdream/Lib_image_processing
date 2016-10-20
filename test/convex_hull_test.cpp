#include <iostream>
#include <pcl/surface/convex_hull.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <image_processing/tools.hpp>
#include <image_processing/pcl_types.h>


int main(int argc, char** argv){

    if(argc < 2){
        std::cout << "usage one path for pcd file" << std::endl;
        return 1;
    }

    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    reader.read(argv[1], *cloud);

    pcl::PassThrough<pcl::PointXYZRGBA> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, 1.1);
    pass.filter (*cloud);

    std::vector<Eigen::Vector3d> vertices;
    image_processing::tools::extract_convex_hull/*<pcl::PointXYZ>*/(cloud,vertices);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::ConvexHull<pcl::PointXYZ> hull;
//    hull.setInputCloud(cloud);
//    hull.reconstruct(*cloud_hull);

    for(auto it = vertices.begin(); it != vertices.end(); ++it)
        cloud_hull->points.push_back(pcl::PointXYZ((*it)[0],(*it)[1],(*it)[2]));

    std::cout << "input cloud size " << cloud->size() << std::endl;
    std::cout << "hull egdes size " << cloud_hull->size() << std::endl;

    pcl::visualization::CloudViewer viewer("cloud");

    viewer.showCloud(cloud_hull);

    while(!viewer.wasStopped());

    return 0;
}
