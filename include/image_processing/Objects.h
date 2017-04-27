#ifndef _OBJECTS_H
#define _OBJECTS_H

#include <iterator>

#include "image_processing/pcl_types.h"

namespace image_processing{

class Blob{
public:
  Blob(PointCloudT blob_cloud) :
    blob_cloud(blob_cloud), children(std::vector<Blob>()) {}

  Blob(const Blob& blob) :
    blob_cloud(blob.blob_cloud), children(blob.children) {}

  ~Blob() {}

  void add_child(Blob& blob);

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

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version){
      ar & objects;
  }

  std::vector<Blob> objects;
};

}

#endif //_OBJECTS_H
