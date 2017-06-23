#ifndef _OBJECT_H
#define _OBJECT_H

#include <iterator>
#include <cstdio>
#include <limits>
#include <cmath>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include <pcl/common/transforms.h>

#include "image_processing/tools.hpp"
#include "image_processing/pcl_types.h"
#include "image_processing/pcl_serialization.h"
#include "image_processing/SurfaceOfInterest.h"


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

  Object(classifier_t classifier, std::string saliency_modality,
         std::string modality, Eigen::Vector4d center) :
    _classifier(classifier), _saliency_modality(saliency_modality),
    _modality(modality), _center(center) {}

  Object(const Object& obj) :
    _classifier(obj._classifier), _saliency_modality(obj._saliency_modality),
    _modality(obj._modality), _center(obj._center) {}

  ~Object() {}

  classifier_t get_classifier() {return _classifier;}

  std::string get_saliency_modality() {return _saliency_modality;}

  std::string get_modality() {return _modality;}

  Eigen::Vector4d get_center() {return _center;}

  void set_initial(SurfaceOfInterest& initial_surface);

  PointCloudT::Ptr get_initial_cloud() {return _initial_cloud;}

  void set_current(SurfaceOfInterest& current_surface,
                   Eigen::Affine3f& transformation);

  PointCloudT::Ptr get_transformed_initial_cloud() {return _transformed_initial_cloud;}

  PointCloudT::Ptr get_current_cloud() {return _current_cloud;}

private:

  classifier_t _classifier;
  std::string _saliency_modality;
  std::string _modality;
  Eigen::Vector4d _center;

  SupervoxelArray _initial_hyp;
  saliency_map_t _initial_map;
  PointCloudT::Ptr _initial_cloud;

  SupervoxelArray _current_hyp;
  saliency_map_t _current_map;
  PointCloudT::Ptr _transformed_initial_cloud;
  PointCloudT::Ptr _current_cloud;
};

template<typename classifier_t>
void Object<classifier_t>::set_initial(SurfaceOfInterest& initial_surface)
{
  _initial_map = initial_surface.compute_saliency_map(_modality, _classifier);

  std::vector<uint32_t> initial_region = initial_surface.get_region_at(_saliency_modality, 0.5, _center);

  SupervoxelArray svs = initial_surface.getSupervoxels();
  _initial_hyp.clear();
  _initial_cloud = PointCloudT::Ptr(new PointCloudT);
  for (const auto& label : initial_region)
  {
    _initial_hyp[label] = svs[label];
    for (const auto& pt : *(svs[label]->voxels_))
    {
      PointT new_pt(pt);
      _initial_cloud->push_back(new_pt);
    }
  }
}

template<typename classifier_t>
void Object<classifier_t>::set_current(SurfaceOfInterest& current_surface,
                                       Eigen::Affine3f& transformation)
{
  // set transformation
  Eigen::Vector4f c;
  Eigen::Affine3f trans = Eigen::Affine3f::Identity();
  pcl::compute3DCentroid<PointT>(*_initial_cloud, c);
  trans.translation().matrix() = Eigen::Vector3f(c[0], c[1], c[2]);

  transformation = transformation * trans.inverse();

  // update transformed initial cloud
  _transformed_initial_cloud = PointCloudT::Ptr(new PointCloudT);
  pcl::transformPointCloud<PointT>(*_initial_cloud, *_transformed_initial_cloud, transformation);
  _current_map = current_surface.compute_saliency_map(_modality, _classifier);

  SupervoxelArray svs = current_surface.getSupervoxels();
  _current_hyp.clear();
  _current_cloud = PointCloudT::Ptr(new PointCloudT);
  for (const auto& initial_sv : _initial_hyp)
  {
    uint32_t min_label = 0;
    double min_d = std::numeric_limits<double>::max();

    // transform the initial supervoxel
    PointCloudT transformed_initial_sv;
    pcl::transformPointCloud<PointT>(*(initial_sv.second->voxels_), transformed_initial_sv, transformation);
    Eigen::Vector4d center;
    pcl::compute3DCentroid<PointT>(transformed_initial_sv, center);

    // find closest current supervoxels in image plan
    for (const auto& current_sv : svs)
    {
      double dx = center[0] - current_sv.second->centroid_.x;
      double dy = center[1] - current_sv.second->centroid_.y;

      double d = sqrt(dx*dx + dy*dy);
      if (d < min_d) {
        min_label = current_sv.first;
        min_d = d;
      }
    }

    pcl::Supervoxel<PointT>::Ptr min_sv = svs[min_label];

    // check for occlusions
    double dz = center[2] - min_sv->centroid_.z;
    if (dz < 0.02) {

      // check coherence
      double coherence = tools::cloud_distance(transformed_initial_sv, *(min_sv->voxels_));
      std::cout << "sv coherence : " << coherence << '\n';
      if (coherence < 0.01) {
        Eigen::VectorXd features = current_surface.get_feature(initial_sv.first, _modality);
        _classifier.add(features, 1);

        _current_hyp[min_label] = min_sv;
        for (const auto& pt : *(min_sv->voxels_))
        {
          PointT new_pt;
          new_pt.x = pt.x;
          new_pt.y = pt.y;
          new_pt.z = pt.z;
          new_pt.r = 0;
          new_pt.g = 255;
          new_pt.b = 0;
          _current_cloud->push_back(new_pt);
        }
      }
      else {
        Eigen::VectorXd features = current_surface.get_feature(initial_sv.first, _modality);
        _classifier.add(features, 0);

        for (const auto& pt : *(min_sv->voxels_))
        {
          PointT new_pt;
          new_pt.x = pt.x;
          new_pt.y = pt.y;
          new_pt.z = pt.z;
          new_pt.r = 255;
          new_pt.g = 0;
          new_pt.b = 0;
          _current_cloud->push_back(new_pt);
        }
      }
    }
    else {
      for (const auto& pt : *(min_sv->voxels_))
      {
        PointT new_pt;
        new_pt.x = pt.x;
        new_pt.y = pt.y;
        new_pt.z = pt.z;
        new_pt.r = 255;
        new_pt.g = 255;
        new_pt.b = 0;
        _current_cloud->push_back(new_pt);
      }
    }
  }
}

}

#endif //_OBJECT_H
