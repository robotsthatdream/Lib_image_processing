#include "image_processing/DescriptorExtraction.h"

#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

using namespace image_processing;

DescriptorExtraction::DescriptorExtraction(const cv::Mat &input, const std::string &type)
  : _input_image(input),
    _type(type)
{
    init<parameters::supervoxel>();
    if(_type.compare("SIFT") == 0){
      _detector = new cv::SiftFeatureDetector();
      _extractor = new cv::SiftDescriptorExtractor();
    }else if(_type.compare("ORB") == 0){
      _detector = new cv::OrbFeatureDetector();
      _extractor = new cv::OrbDescriptorExtractor();
    }else{
      _detector = cv::FeatureDetector::create(_type);
      _extractor = cv::DescriptorExtractor::create(_type);
    }

}

void DescriptorExtraction::detect(){
  _detector->detect(_input_image,_key_points);
}



void DescriptorExtraction::extract(){
  _extractor->compute(_input_image,_key_points,_descriptors);

}

void DescriptorExtraction::kmeans_clustering(int K){


  cv::Mat points(_key_points.size(),2,CV_32F);
  for(int i = 0; i < _key_points.size(); i++ ){
      points.at<float>(i,0) = _key_points[i].pt.x;
      points.at<float>(i,1) = _key_points[i].pt.y;
    }
  std::vector<int> labels;
  cv::Mat centers;
  cv::kmeans(points,K,labels,cv::TermCriteria(cv::TermCriteria::COUNT,1000,0.01),0,cv::KMEANS_PP_CENTERS,centers);
  std::vector<cv::KeyPoint> clustered_keypoints(K);
  std::vector<float> vct(K);
  for(int i = 0; i < clustered_keypoints.size(); i++){
      clustered_keypoints[i].angle = 0;
      clustered_keypoints[i].size = 0;
      clustered_keypoints[i].response = 0;
      clustered_keypoints[i].octave = 0;
      vct[i] = 0;
    }

  for(int i = 0; i < labels.size(); i++){
      vct[labels[i]] += 1;
      clustered_keypoints[labels[i]].angle = clustered_keypoints[labels[i]].angle
          + (_key_points[i].angle - clustered_keypoints[labels[i]].angle )/vct[labels[i]];
      clustered_keypoints[labels[i]].size = clustered_keypoints[labels[i]].size
          + (_key_points[i].size - clustered_keypoints[labels[i]].size)/vct[labels[i]];
      clustered_keypoints[labels[i]].response = clustered_keypoints[labels[i]].response
          + (_key_points[i].response - clustered_keypoints[labels[i]].response )/vct[labels[i]];
      clustered_keypoints[labels[i]].octave = _key_points[i].octave; /*clustered_keypoints[labels[i]].octave
          + (_key_points[i].octave - clustered_keypoints[labels[i]].octave )/vct[labels[i]];*/
      clustered_keypoints[labels[i]].pt.x = centers.at<float>(labels[i],0);
      clustered_keypoints[labels[i]].pt.y = centers.at<float>(labels[i],1);
    }

  _key_points = clustered_keypoints;

}


void DescriptorExtraction::align(const PointCloudT &ptcl){
    _3d_positions.clear();
    for(int i = 0; i < _key_points.size(); i++){
        cv::KeyPoint currentPt = _key_points[i];

        PointT pt = ptcl.at((int) currentPt.pt.x+(int)currentPt.pt.y*ptcl.width);

        float depth_value = pt.z;
        float x_value = (currentPt.pt.x-_param.rgb_princ_pt_x)*depth_value/_param.focal_length_x;
        float y_value = (currentPt.pt.y-_param.rgb_princ_pt_y)*depth_value/_param.focal_length_y;
        if(!boost::math::isnan<float>(x_value) && !boost::math::isnan<float>(y_value) && !boost::math::isnan<float>(depth_value)){
            pcl::PointXYZ pt3d(x_value ,y_value,depth_value);
            Feature f;
            f.point = pt3d;
            f.keypoint = currentPt;
            f.descriptor = _descriptors.row(i);
            _features.push_back(f);
            _3d_positions.push_back(pt3d);
          }
      }
}

DescriptorExtraction::KeyPointsXYZ DescriptorExtraction::pcl_key_points(const PointCloudT &ptcl, float radius){

  KeyPointsXYZ key_points;

  for(int i = 0; i < _key_points.size(); i++){
      cv::KeyPoint currentPt = _key_points[i];

      PointT pt = ptcl.at((int) currentPt.pt.x+(int)currentPt.pt.y*ptcl.width);
      float x = pt.x;
      float y = pt.y;
      float depth_value = pt.z;
      float x_value = (currentPt.pt.x-_param.rgb_princ_pt_x)*depth_value/_param.focal_length_x;
      float y_value = (currentPt.pt.y-_param.rgb_princ_pt_y)*depth_value/_param.focal_length_y;
      double distance = sqrt((depth_value-0.85)*(depth_value-0.85) + (x_value-0.35)*(x_value-0.35) + (y_value-0.4)*(y_value-0.4));
      if(!boost::math::isnan<float>(x_value) && !boost::math::isnan<float>(y_value) && !boost::math::isnan<float>(depth_value)
         && distance < radius){
          pcl::PointXYZ* pt3d = new pcl::PointXYZ(x_value ,y_value,depth_value);
          key_points.insert(std::pair<pcl::PointXYZ*,cv::KeyPoint>(pt3d,currentPt));
          _3d_positions.push_back(*pt3d);
        }
    }
  return key_points;
}

std::vector<double> DescriptorExtraction::match(const cv::Mat &descritors1, const cv::Mat &descritors2){

  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
  std::vector<cv::DMatch> matches;
  matcher->match(descritors1,descritors2,matches);
  std::vector<double> res;
  for(int i = 0; i < matches.size(); i++)
    res.push_back(matches[i].distance);

  return res;
}

void DescriptorExtraction::save_descriptors(){
  //    for(int i = 0; i < SIFT_descriptors.size(); i++){
  //        std::stringstream ss;
  //        ss << "data/sift/sift_descr_" << counter << "_" << i <<  ".png";
  //        if(!cv::imwrite(ss.str(),SIFT_descriptors[i])){
  //            std::cerr << "error cv::imwrite"  << std::endl;
  //            return;
  //        }
  //    }
}



void DescriptorExtraction::descriptor_clustering(){
  //    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
  //    for(int i = 0; i < SIFT_descriptors.size(); i++){
  //        for(int j = 0; j < SIFT_descriptors.size(); j++){
  //            if(i == j)
  //                continue;
  //            matcher->match(SIFT_descriptors[i],SIFT_descriptors[j]);

  //        }
  //    }

}

void DescriptorExtraction::set_type(const std::string& t){
  _type = t;
  if(_type.compare("SIFT") == 0){
      _detector = new cv::SiftFeatureDetector();
      _extractor = new cv::SiftDescriptorExtractor();
    }else if(_type.compare("ORB") == 0){
      _detector = new cv::OrbFeatureDetector();
      _extractor = new cv::OrbDescriptorExtractor();
    }else{
      _detector = cv::FeatureDetector::create(_type);
      _extractor = cv::DescriptorExtractor::create(_type);
    }
}

void DescriptorExtraction::clear(){
  _key_points.clear();
  _features.clear();
  _3d_positions.clear();
}
