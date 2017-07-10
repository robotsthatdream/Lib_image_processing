#ifndef _OBJECT_H
#define _OBJECT_H

#include <iterator>
#include <cstdio>
#include <limits>
#include <cmath>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/octree/octree_search.h>
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

  void set_center(Eigen::Vector4d center) {_center = center;}

  Eigen::Vector4d get_center() {return _center;}

  void recover_center(SurfaceOfInterest& surface);

  bool set_initial(SurfaceOfInterest& initial_surface);

  PointCloudT::Ptr get_initial_cloud() {return _initial_cloud;}

  bool set_current(SurfaceOfInterest& current_surface,
                   Eigen::Affine3f& transformation);

  PointCloudT::Ptr get_transformed_initial_cloud() {return _transformed_initial_cloud;}

  PointCloudT::Ptr get_current_cloud() {return _current_cloud;}

  PointCloudT::Ptr get_result_cloud() {return _result_cloud;}

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
  PointCloudT::Ptr _result_cloud;
};

template<typename classifier_t>
void Object<classifier_t>::recover_center(SurfaceOfInterest& surface)
{
  std::cout << "object hypothesis : recovering object's center" << std::endl;

  saliency_map_t map = surface.compute_saliency_map(_modality, _classifier);

  uint32_t max_label = 0;
  double max_saliency = 0;
  for (const auto& entry : map)
  {
    if (entry.second > max_saliency) {
      max_label = entry.first;
      max_saliency = entry.second;
    }
  }

  if (max_saliency > 0) {
    SupervoxelArray svs = surface.getSupervoxels();
    pcl::compute3DCentroid(*(svs[max_label]->voxels_), _center);
  }
  else {
    std::cerr << "object hypothesis : object's center could not be recovered" << std::endl;
  }
}

template<typename classifier_t>
bool Object<classifier_t>::set_initial(SurfaceOfInterest& initial_surface)
{
  _initial_hyp.clear();
  _initial_cloud = PointCloudT::Ptr(new PointCloudT);
  _initial_map = initial_surface.compute_saliency_map(_modality, _classifier);

  std::vector<uint32_t> initial_region = initial_surface.get_region_at(_saliency_modality, 0.5, _center);

  if (initial_region.size() == 0) {
    std::cerr << "object hypothesis : initial region empty" << std::endl;
    return false;
  }

  SupervoxelArray svs = initial_surface.getSupervoxels();
  for (const auto& label : initial_region)
  {
    _initial_hyp[label] = svs[label];
    for (const auto& pt : *(svs[label]->voxels_))
    {
      PointT new_pt(pt);
      _initial_cloud->push_back(new_pt);
    }
  }
  pcl::compute3DCentroid<PointT>(*_initial_cloud, _center);

  // supervoxels that do not belong to the hypothesis are negative examples
  std::vector<Eigen::VectorXd> features(0);
  std::vector<int> labels(0);
  for (const auto& sv : svs)
  {
    if (_initial_hyp.count(sv.first) == 0) {
      features.push_back(initial_surface.get_feature(sv.first, _modality));
      labels.push_back(0);
    }
  }
  _classifier.fit_batch(features, labels);

  return true;
}

template<typename classifier_t>
bool Object<classifier_t>::set_current(SurfaceOfInterest& current_surface,
                                       Eigen::Affine3f& transformation)
{
  _transformed_initial_cloud = PointCloudT::Ptr(new PointCloudT);

  _current_hyp.clear();
  _current_cloud = PointCloudT::Ptr(new PointCloudT);
  _current_map = current_surface.compute_saliency_map(_modality, _classifier);

  _result_cloud = PointCloudT::Ptr(new PointCloudT);

  // set transformation (transformation's origin is the center of initial cloud)
  Eigen::Vector4d c_initial;
  Eigen::Affine3f trans = Eigen::Affine3f::Identity();
  pcl::compute3DCentroid<PointT>(*_initial_cloud, c_initial);
  trans.translation().matrix() = Eigen::Vector3f(c_initial[0], c_initial[1], c_initial[2]);

  transformation = transformation * trans.inverse();

  // first update of transformed initial cloud
  pcl::transformPointCloud<PointT>(*_initial_cloud, *_transformed_initial_cloud, transformation);
  Eigen::Vector4d c_transformed_initial;
  pcl::compute3DCentroid(*_transformed_initial_cloud, c_transformed_initial);

  // set current hypothesis, cloud and map
  SupervoxelArray svs = current_surface.getSupervoxels();
  std::vector<uint32_t> current_region = current_surface.get_region_at(_saliency_modality, 0.5, c_transformed_initial);
  if (current_region.size() == 0) {
    std::cerr << "object hypothesis : current region empty" << std::endl;
    return false;
  }
  for (const auto& label : current_region)
  {
    _current_hyp[label] = svs[label];
    for (const auto& pt : *(svs[label]->voxels_))
    {
      PointT n_pt(pt);
      _current_cloud->push_back(n_pt);
    }
  }

  // align cloud to improve the correspondance
  PointCloudT aligned_initial_cloud;
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setInputCloud(_transformed_initial_cloud);
  icp.setInputTarget(_current_cloud);
  icp.setMaximumIterations(25);
  icp.setTransformationEpsilon(1e-9);
  icp.setMaxCorrespondenceDistance(0.05);
  icp.setEuclideanFitnessEpsilon(1);

  icp.align(aligned_initial_cloud);

  // tracking has failed ?
  if (!icp.hasConverged()){
    std::cerr << "object hypothesis : tracking might have failed (icp has not converged)" << std::endl;
    return false;
  }

  Eigen::Affine3f icp_trans;
  icp_trans.matrix() = icp.getFinalTransformation();

  transformation = icp_trans * transformation;

  // update the current center of the object
  _transformed_initial_cloud = PointCloudT::Ptr(new PointCloudT);
  pcl::transformPointCloud<PointT>(*_initial_cloud, *_transformed_initial_cloud, transformation);
  pcl::compute3DCentroid(*_transformed_initial_cloud, c_transformed_initial);

  // set coherence and search methods
  DistanceCoherence<PointT> distance_coherence;
  HSVColorCoherence<PointT> color_coherence;
  // NormalCoherence<PointT> normal_coherence;

  pcl::octree::OctreePointCloudSearch<PointT> octree(0.001);
  octree.setInputCloud(_current_cloud);
  octree.addPointsFromInputCloud();

  // gathering training samples
  std::vector<Eigen::VectorXd> samples(0);
  std::vector<int> labels(0);
  int n_pos = 0;
  int n_neg = 0;
  for (const auto& initial_sv : _initial_hyp)
  {
    // transform the initial supervoxel
    PointCloudT::Ptr transformed_initial_sv = PointCloudT::Ptr(new PointCloudT);
    pcl::transformPointCloud<PointT>(*(initial_sv.second->voxels_), *transformed_initial_sv, transformation);

    // compute coherence
    double w = 0;
    int nb_pt = 0;
    for (auto& pt : *transformed_initial_sv)
    {
      std::vector<int> point_idx;
      std::vector<float> point_distance;
      octree.nearestKSearch(pt, 1, point_idx, point_distance);

      if (point_distance[0] < 0.005) {
        double w_pt = 1;
        w_pt *= distance_coherence.compute(pt, _current_cloud->points[point_idx[0]]);
        w_pt *= color_coherence.compute(pt, _current_cloud->points[point_idx[0]]);
        // w_pt *= normal_coherence.compute(pt, _current_cloud->points[point_idx[0]]);

        w += w_pt;
        nb_pt += 1;
      }
    }
    w /= nb_pt;

    // check occlusion
    if (transformed_initial_sv->size() > 2*nb_pt) {
      w = -1;
    }

    // check coherence;
    std::cout << "sv coherence : " << w << "(" << nb_pt << "/" << transformed_initial_sv->size() << ")" << std::endl;
    if (w > 0.90) {
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
        _result_cloud->push_back(new_pt);
      }
    }
    else if (w >= 0) {
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
        _result_cloud->push_back(new_pt);
      }
    }
    else {
      for (const auto& pt : *transformed_initial_sv)
      {
        PointT new_pt;
        new_pt.x = pt.x;
        new_pt.y = pt.y;
        new_pt.z = pt.z;
        new_pt.r = 255;
        new_pt.g = 255;
        new_pt.b = 0;
        _result_cloud->push_back(new_pt);
      }
    }
  }

  // the object has moved ?
  Eigen::Vector4d c_current;
  pcl::compute3DCentroid(*_transformed_initial_cloud, c_current);
  if ((c_initial - c_current).norm() < 0.03) {
    std::cerr << "object hypothesis : object mouvement is less then 3 cm" << std::endl;
    return false;
  }

  _center = c_current;

  // the tracking did not fail ?
  if (n_pos < n_neg) {
    std::cerr << "object hypothesis : tracking might have failed (n_pos=" << n_pos << ", n_neg=" << n_neg << ")" << std::endl;
    return false;
  }

  // update model
  _classifier.fit_batch(samples, labels);

  return true;
}

}

#endif //_OBJECT_H
