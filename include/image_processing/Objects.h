#ifndef _OBJECTS_H
#define _OBJECTS_H

#include <iterator>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include <pcl/common/transforms.h>

#include "image_processing/tools.hpp"
#include "image_processing/pcl_types.h"
#include "image_processing/pcl_serialization.h"
#include "image_processing/SupervoxelSet.h"


namespace image_processing{

// class Object{
// public:
//   Object() {}
//
//   Object(PointCloudT object_cloud) :
//     object_cloud(object_cloud) {}
//
//   Object(const Object& object) :
//     object_cloud(object.object_cloud), features(object.features) {}
//
//   ~Object() {}
//
//   void update_cloud(PointCloudT current_cloud);
//
//   friend class boost::serialization::access;
//
//   template<class Archive>
//   void serialize(Archive& ar, const unsigned int version){
//       ar & object_cloud & features;
//   }
//
//   PointCloudT object_cloud;
//   std::vector<std::vector<double>> features;
// };


template<class classifier_t>
class Object{
public:

  Object() {}

  Object(classifier_t classifier, features_extractor_t features_extractor) :
    _classifier(classifier), _features_extractor(features_extractor) {}

  Object(const Object& obj) :
    _classifier(obj._classifier), _features_extractor(obj._features_extractor) {}

  ~Object() {}

  classifier_t get_classifier() {return _classifier;}

  SaliencyMap get_saliency_map(SupervoxelSet& supervoxels);

  void set_initial(SupervoxelSet& initial_supervoxels,
                   uint32_t initial_label);

  PointCloudT::Ptr get_initial_cloud() {return _initial_cloud;}

  void set_current(SupervoxelSet& current_supervoxels,
                   Eigen::Affine3f& transformation);

  PointCloudT::Ptr get_transformed_initial_cloud() {return _transformed_initial_cloud;}

  PointCloudT::Ptr get_current_cloud() {return _current_cloud;}

private:

  classifier_t _classifier;
  features_extractor_t _features_extractor;

  SupervoxelArray _initial_hyp;
  PointCloudT::Ptr _initial_cloud;

  SupervoxelArray _current_hyp;
  PointCloudT::Ptr _transformed_initial_cloud;
  PointCloudT::Ptr _current_cloud;
};


template<typename classifier_t>
SaliencyMap Object<classifier_t>::get_saliency_map(SupervoxelSet& supervoxels)
{
  SaliencyMap weights;

  SupervoxelArray svs = supervoxels.getSupervoxels();
  for (const auto& sv : svs)
  {
    Eigen::VectorXd features = _features_extractor(sv.second);
    weights[sv.first] = _classifier.compute_estimation(features, 1);
  }

  return weights;
}

template<typename classifier_t>
void Object<classifier_t>::set_initial(SupervoxelSet& initial_supervoxels,
                                          uint32_t initial_label)
{
  SupervoxelArray hyp;

  SupervoxelArray svs = initial_supervoxels.getSupervoxels();
  std::vector<uint32_t> labels = initial_supervoxels.getNeighbor(initial_label);

  hyp[initial_label] = svs[initial_label];
  for (size_t i = 0; i < labels.size(); i++)
  {
    hyp[labels[i]] = svs[labels[i]];
  }

  _initial_hyp = hyp;

  // update initial cloud
  _initial_cloud = PointCloudT::Ptr(new PointCloudT);
  for (const auto& sv : _initial_hyp)
  {
    for (const auto& pt : *(sv.second->voxels_))
    {
      PointT new_pt;
      new_pt.x = pt.x;
      new_pt.y = pt.y;
      new_pt.z = pt.z;
      new_pt.r = pt.r;
      new_pt.g = pt.g;
      new_pt.b = pt.b;
      _initial_cloud->push_back(new_pt);
    }
  }
}

template<typename classifier_t>
void Object<classifier_t>::set_current(SupervoxelSet& current_supervoxels,
                                          Eigen::Affine3f& transformation)
{
  // update transformed initial cloud
  _transformed_initial_cloud = PointCloudT::Ptr(new PointCloudT);
  pcl::transformPointCloud<PointT>(*_initial_cloud, *_transformed_initial_cloud, transformation);

  SupervoxelArray hyp;

  SupervoxelArray svs = current_supervoxels.getSupervoxels();
  uint32_t min_label = 0;
  double min_distance = std::numeric_limits<double>::max();
  for (const auto& initial_sv : _initial_hyp)
  {
    // find closest current supervoxels in image plan
    for (const auto& current_sv : svs)
    {
      double dx = initial_sv.second->centroid_.x - current_sv.second->centroid_.x;
      double dy = initial_sv.second->centroid_.y - current_sv.second->centroid_.y;

      double d = sqrt(dx*dx + dy*dy);
      if (d < min_distance) {
        min_label = current_sv.first;
        min_distance = d;
      }
    }

    pcl::Supervoxel<PointT>::Ptr min_sv = svs[min_label];

    // check for occlusions
    double dz = initial_sv.second->centroid_.z - min_sv->centroid_.z;
    if (dz < 0.01) {

      // check coherence
      PointCloudT transformed_initial_sv;
      pcl::transformPointCloud<PointT>(*(initial_sv.second->voxels_), transformed_initial_sv, transformation);

      double coherence = tools::cloud_distance(transformed_initial_sv, *(min_sv->voxels_));
      if (coherence < 0.01) {
        Eigen::VectorXd features = _features_extractor(initial_sv.second);
        _classifier.add(features, 1);

        hyp[min_label] = min_sv;
      }
      else {
        Eigen::VectorXd features = _features_extractor(initial_sv.second);
        _classifier.add(features, 0);
      }
    }
  }

  _current_hyp = hyp;

  // update current cloud
  _current_cloud = PointCloudT::Ptr(new PointCloudT);
  for (const auto& sv : _current_hyp)
  {
    for (const auto& pt : *(sv.second->voxels_))
    {
      PointT new_pt;
      new_pt.x = pt.x;
      new_pt.y = pt.y;
      new_pt.z = pt.z;
      new_pt.r = pt.r;
      new_pt.g = pt.g;
      new_pt.b = pt.b;
      _initial_cloud->push_back(new_pt);
    }
  }
}

}

#endif //_OBJECTS_H
