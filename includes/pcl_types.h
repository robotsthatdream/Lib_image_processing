#ifndef PCL_TYPES_H
#define PCL_TYPES_H

#include <pcl/point_types.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <map>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<pcl::Normal> PointCloudN;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;
typedef std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr > SupervoxelArray;
typedef std::multimap<uint32_t,uint32_t> AdjacencyMap;


#endif //PCL_TYPES_H
