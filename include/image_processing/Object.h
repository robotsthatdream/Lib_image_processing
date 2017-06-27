#ifndef _OBJECT_H
#define _OBJECT_H

#include <iterator>
#include <cstdio>
#include <limits>
#include <cmath>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include <pcl/common/transforms.h>
#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/hsv_color_coherence.h>
// #include <pcl/tracking/normal_coherence.h>

#include "image_processing/tools.hpp"
#include "image_processing/pcl_types.h"
#include "image_processing/pcl_serialization.h"
#include "image_processing/SurfaceOfInterest.h"

using namespace pcl::tracking;

namespace image_processing{

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

  class _Coherence : public NearestPairPointCloudCoherence<PointT>
  {
  public:

    bool initCompute()
    {
      return NearestPairPointCloudCoherence::initCompute();
    }

    void computeCoherence (const PointCloudT::Ptr &cloud, const pcl::IndicesPtr &indices, float &w_j)
    {
      NearestPairPointCloudCoherence::computeCoherence(cloud, indices, w_j);
    }

  };

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
  // set transformation (transformation's origin is the center of initial cloud)
  Eigen::Vector4f c_initial;
  Eigen::Affine3f trans = Eigen::Affine3f::Identity();
  pcl::compute3DCentroid<PointT>(*_initial_cloud, c_initial);
  trans.translation().matrix() = Eigen::Vector3f(c_initial[0], c_initial[1], c_initial[2]);

  transformation = transformation * trans.inverse();


  // set target cloud
  PointCloudT::Ptr target_cloud = PointCloudT::Ptr(new PointCloudT);
  SupervoxelArray svs = current_surface.getSupervoxels();
  for (const auto& sv : svs)
  {
    for (const auto& pt : *(sv.second->voxels_))
    {
      PointT n_pt(pt);
      target_cloud->push_back(n_pt);
    }
  }


  // set coherence
  _Coherence coherence;

  boost::shared_ptr<DistanceCoherence<PointT> > distance_coherence
    = boost::shared_ptr<DistanceCoherence<PointT> >(new DistanceCoherence<PointT>());
  coherence.addPointCoherence(distance_coherence);

  boost::shared_ptr<HSVColorCoherence<PointT> > color_coherence
    = boost::shared_ptr<HSVColorCoherence<PointT> >(new HSVColorCoherence<PointT>());
  coherence.addPointCoherence(color_coherence);

  // boost::shared_ptr<NormalCoherence<PointT> > normal_coherence
  //   = boost::shared_ptr<NormalCoherence<PointT> >(new NormalCoherence<PointT>());
  // coherence.addPointCoherence(normal_coherence);

  boost::shared_ptr<pcl::search::Octree<PointT> > search (new pcl::search::Octree<PointT>(0.01));
  coherence.setSearchMethod(search);
  coherence.setMaximumDistance(0.01);
  coherence.setTargetCloud(target_cloud);
  coherence.initCompute();


  // update transformed initial cloud
  _transformed_initial_cloud = PointCloudT::Ptr(new PointCloudT);
  pcl::transformPointCloud<PointT>(*_initial_cloud, *_transformed_initial_cloud, transformation);
  _current_map = current_surface.compute_saliency_map(_modality, _classifier);


  // gathering training samples
  _current_hyp.clear();
  _current_cloud = PointCloudT::Ptr(new PointCloudT);
  std::vector<Eigen::VectorXd> samples;
  std::vector<uint32_t> labels;
  int n_pos = 0;
  int n_neg = 0;
  for (const auto& initial_sv : _initial_hyp)
  {
    uint32_t min_label = 0;
    double min_d = std::numeric_limits<double>::max();

    // transform the initial supervoxel
    PointCloudT::Ptr transformed_initial_sv = PointCloudT::Ptr(new PointCloudT);
    pcl::transformPointCloud<PointT>(*(initial_sv.second->voxels_), *transformed_initial_sv, transformation);
    Eigen::Vector4d center;
    pcl::compute3DCentroid<PointT>(*transformed_initial_sv, center);

    float coherence_w = 0;
    pcl::IndicesPtr indices;
    coherence.computeCoherence(transformed_initial_sv, indices, coherence_w);
    coherence_w /= transformed_initial_sv->size();

    // check coherence;
    std::cout << "sv coherence : " << coherence_w << '\n';
    if (coherence_w < - 0.90) {
      Eigen::VectorXd features = current_surface.get_feature(initial_sv.first, _modality);
      samples.push_back(features);
      labels.push_back(1);
      n_pos += 1;

      for (const auto& pt : *transformed_initial_sv)
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
      samples.push_back(features);
      labels.push_back(0);
      n_neg += 1;

      for (const auto& pt : *transformed_initial_sv)
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

  // the object has moved ?
  Eigen::Vector4f c_current;
  pcl::compute3DCentroid<PointT>(*_transformed_initial_cloud, c_current);
  if ((c_initial - c_current).norm() < 0.03) {
    std::cerr << "image processing : object mouvement is less then 3 cm" << std::endl;
    return;
  }

  // the tracking did not fail ?
  if (n_pos < n_neg) {
    std::cerr << "image processing : tracking might have failed (n_pos=" << n_pos << ", n_neg=" << n_neg << ")" << std::endl;
    return;
  }

  // update model
  for (size_t i = 0; i < samples.size(); i++)
  {
    _classifier.fit(samples[i], labels[i]);
  }

  return;
}

}

#endif //_OBJECT_H
