#include <iostream>
#include <pcl_types.h>
#include <Eigen/Core>
#include <pcl/surface/convex_hull.h>

namespace image_processing{

static bool extract_convex_hull(const PointCloudT::Ptr cloud,  std::vector<Eigen::Vector3d>& vertices){
      pcl::ConvexHull hull_extractor;
      PointCloudXYZ hull_cloud;
      hull_extractor.setInputCloud(cloud);
      hull_extractor.reconstruct(hull_cloud);

      if(hull_cloud.empty()){
          std::cerr << "unable to compute the convex hull" << std::endl;
          return false;
      }

      for(auto it = hull_cloud.points.begin(); it != hull_cloud.points.end(); ++it)
          vertices.push_back(Eigen::Vector3d(it->x,it->y,it->z));


      return true;
}
}
