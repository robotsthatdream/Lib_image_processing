#ifndef _OBJECTS_H
#define _OBJECTS_H

#include <iterator>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "image_processing/pcl_types.h"
#include "image_processing/pcl_serialization.h"

namespace image_processing{

class Blob{
public:
  Blob() {}

  Blob(PointCloudT blob_cloud) :
    blob_cloud(blob_cloud), children(std::vector<Blob>()) {}

  Blob(const Blob& blob) :
    blob_cloud(blob.blob_cloud), children(blob.children) {}

  ~Blob() {}

  void add_child(Blob& blob);

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version){
      ar & blob_cloud & children;
  }

  PointCloudT blob_cloud;
  std::vector<Blob> children;
  // add classifier or model ??
};

class Objects{
public:
  Objects() : objects(std::vector<Blob>()) {}

  Objects(const Objects& objects) :
    objects(objects.objects) {}

  ~Objects() {}

  void add(PointCloudT blob_cloud);

  void remove(std::vector<Blob>::iterator position);

  void merge(std::vector<Blob>::iterator pos1, std::vector<Blob>::iterator pos2,
    PointCloudT parent_cloud);

  void split(std::vector<Blob>::iterator pos);

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version){
      ar & objects;
  }

  std::vector<Blob> objects;
};

}

#endif //_OBJECTS_H
