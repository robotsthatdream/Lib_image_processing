#ifndef PCL_TYPES_H
#define PCL_TYPES_H

#include <pcl/point_types.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <opencv2/opencv.hpp>
#include <map>

namespace image_processing{

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointXYZHSV PointHSV;
typedef pcl::PointCloud<PointHSV> PointCloudHSV;
typedef pcl::PointCloud<pcl::Normal> PointCloudN;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;
typedef std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> SupervoxelArray;
typedef std::multimap<uint32_t, uint32_t> AdjacencyMap;
typedef std::map<uint32_t, std::vector<cv::Point2f>> Superpixels;
typedef std::map<uint32_t, double> SaliencyMap;
typedef std::function<Eigen::VectorXd(pcl::Supervoxel<PointT>::Ptr)> features_extractor_t;

}

#endif //PCL_TYPES_H
