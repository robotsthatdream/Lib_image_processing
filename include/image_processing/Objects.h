#ifndef _OBJECTS_H
#define _OBJECTS_H

#include <iterator>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "image_processing/pcl_types.h"
#include "image_processing/pcl_serialization.h"

namespace image_processing{

class Object{
public:
  Object() {}

  Object(PointCloudT object_cloud) :
    object_cloud(object_cloud) {}

  Object(const Object& object) :
    object_cloud(object.object_cloud), features(object.features) {}

  ~Object() {}

  void update_cloud(PointCloudT current_cloud);

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version){
      ar & object_cloud & features;
  }

  PointCloudT object_cloud;
  std::vector<std::vector<double>> features;
};

}

#endif //_OBJECTS_H
