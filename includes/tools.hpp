#ifndef _TOOLS_HPP
#define _TOOLS_HPP

#include <iostream>
#include <pcl_types.h>
#include <Eigen/Core>
#include <pcl/surface/convex_hull.h>

namespace image_processing{

//TO DO : Templatize
//template<typename point>
bool extract_convex_hull(pcl::PointCloud<PointT>::ConstPtr cloud,  std::vector<Eigen::Vector3d>& vertex_list){
      pcl::ConvexHull<PointT> hull_extractor;
      pcl::PointCloud<PointT> hull_cloud;
      hull_extractor.setInputCloud(cloud);
      hull_extractor.reconstruct(hull_cloud);

      if(hull_cloud.empty()){
          std::cerr << "unable to compute the convex hull" << std::endl;
          return false;
      }

      for(auto it = hull_cloud.points.begin(); it != hull_cloud.points.end(); ++it)
          vertex_list.push_back(Eigen::Vector3d(it->x,it->y,it->z));


      return true;
}
}

#endif //_TOOLS_HPP
