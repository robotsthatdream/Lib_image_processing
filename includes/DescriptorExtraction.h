#ifndef DESCRIPTOR_EXTRACTION_H
#define DESCRIPTOR_EXTRACTION_H

#include <opencv2/opencv.hpp>
#include <pcl_types.h>
#include "default_parameters.hpp"

namespace image_processing{

/**
 * @brief The DescriptorExtraction class
 * This class provide all tools to extract visual features. The type of features must be specified before any computation (at the construction or with set_type methodes).
 */
class DescriptorExtraction {

public:
    typedef std::map<pcl::PointXYZ*, cv::KeyPoint> KeyPointsXYZ;

  /**
   * @brief The Feature struct
   * This struct define a feature. It is characterize by it position in space (point), it ketpoint and it descriptor.
   */
    struct Feature{
      pcl::PointXYZ point; /**< a xyz-pcl-point give the position of this position in a pointcloud */
      cv::KeyPoint keypoint; /**< keypoint. computed with opencv */
      cv::Mat descriptor; /**< a one dimensional vertical vector */
    };

    DescriptorExtraction(){init<parameters::camera>();}
    DescriptorExtraction(const cv::Mat& input,const std::string& type);

    template <typename Param>
    void init(){
        _param.depth_princ_pt_x = Param::depth_princ_pt_x;
        _param.depth_princ_pt_y = Param::depth_princ_pt_y;
        _param.focal_length_x = Param::focal_length_x;
        _param.focal_length_y = Param::focal_length_y;
        _param.rgb_princ_pt_x = Param::rgb_princ_pt_x;
        _param.rgb_princ_pt_y = Param::rgb_princ_pt_y;
    }

    /**
     * @brief extract
     * extract the descriptor with input image and it keypoints.
     * this methode must be used after detect methode.
     */
    void extract();
    /**
     * @brief detect
     * detect the keypoints of the input image
     *
     */
    void detect();

    /**
     * @brief kmeans_clustering
     * @param K
     * compute a kmeans spatial clustering on the keypoints and make new keypoints, save those in attribute _key_points (accessible by get_key_points()).
     */
    void kmeans_clustering(int K);

    /**
     * @brief align the computed key points on a 3d pointcloud
     * @param 3d pointcloud
     */
    void align(const PointCloudT &ptcl);

    /**
     * @brief descriptor_clustering TO BE DONE or not
     * useless
     */
    void descriptor_clustering();

    /**
     * @brief save_descriptors TO BE DONE
     */
    void save_descriptors();

    /**
     * @brief match
     * @param descritors1
     * @param descritors2
     * @return list of distances
     *
     * static function which compute the matching distance between two descriptor.
     */
    static std::vector<double> match(const cv::Mat& descritors1, const cv::Mat& descritors2);

    /**
     * @brief pcl_key_points (deprecated)
     * @param ptcl
     * @param radius
     * @return
     *
     * compute the spatial position of the keypoints. this methode must be used after detect methode.
     */
    KeyPointsXYZ pcl_key_points(const PointCloudT& ptcl, float radius = .8);

    //Getters-------------------------------------------------------------------------//

    /**
     * @brief get_key_points_cloud. must called only if compute() was called previously.
     * @return pointcloudXYZ of key points
     */
    const PointCloudXYZ& get_key_points_cloud(){return _3d_positions;}

    /**
     * @brief set_type
     * @param t
     */
    void set_type(const std::string& t);

    /**
     * @brief set_input_image
     * @param img
     */
    void set_input_image(const cv::Mat& img){_input_image = img;}

    /**
     * @brief get_descriptors
     * @return
     */
    cv::Mat get_descriptors(){return _descriptors;}

    /**
     * @brief get_key_points
     * @return
     */
    std::vector<cv::KeyPoint> get_key_points(){return _key_points;}

    /**
     * @brief get_type
     * @return
     */
    std::string get_type(){return _type;}

    /**
     * @brief get_features
     * @return
     */
    std::vector<Feature> get_features(){return _features;}

    void clear();
    //-------------------------------------------------------------------------------//

private:

     std::string _type;

     cv::Mat _input_image;
     cv::Ptr<cv::DescriptorExtractor> _extractor;
     cv::Ptr<cv::FeatureDetector> _detector;

//     pcl::SIFTKeypoint<PointT,PointT> _3d_detector;

     std::vector<cv::KeyPoint> _key_points;
     cv::Mat _descriptors;
     PointCloudXYZ _3d_positions;

     std::vector<Feature> _features;

     struct _param_t{
         float depth_princ_pt_x;
         float depth_princ_pt_y;
         float rgb_princ_pt_x;
         float rgb_princ_pt_y;
         float focal_length_x;
         float focal_length_y;
     };
     _param_t _param;
};

}

#endif //DESCRIPTOR_EXTRACTION_H
