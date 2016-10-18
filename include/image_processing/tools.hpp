#ifndef _TOOLS_HPP
#define _TOOLS_HPP

#include <iostream>
#include "pcl_types.h"
#include <Eigen/Core>
#include <pcl/surface/convex_hull.h>
#include <pcl/tracking/impl/hsv_color_coherence.hpp>

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

void RGB2HSV(const PointCloudT::Ptr input, PointCloudHSV::Ptr output){
    float h, s, v;

    for(auto itr = input->begin(); itr != input->end(); itr++){
        pcl::tracking::RGB2HSV(itr->r,itr->g,itr->b,h,s,v);
        output->push_back(PointHSV(h,s,v));
        output->back().x = itr->x;
        output->back().y = itr->y;
        output->back().z = itr->z;
    }

}

}

#endif //_TOOLS_HPP
