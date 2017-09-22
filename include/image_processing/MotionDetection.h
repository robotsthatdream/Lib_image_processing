#ifndef MOTION_DETECTION_H
#define MOTION_DETECTION_H

#include <opencv2/opencv.hpp>
#include <fstream>
#include <ctime>
#include <image_processing/pcl_types.h>
#include <pcl/octree/octree_pointcloud_changedetector.h>
#include <pcl/segmentation/supervoxel_clustering.h>

namespace image_processing{

/**
 * @brief The MotionDetection class
 * This class provide tools for motion detection based on rgb video camera.
 *
 */
class MotionDetection {

public:
    /**
     * @brief MotionDetection
     * default constructor
     */
    MotionDetection()
    { }

    /**
     * @brief MotionDetection
     * @param vector of 2 successives frames
     * @param minimum size of detection
     */
    MotionDetection(std::vector<cv::Mat> f, double min_area = 100) : _frames(f), _minArea(min_area)
    { }

    /**
     * @deprecated replaced by detect_simple and detect_MOG
     * @brief detects motions as ROIs in a frame.
     * @param image with the contours of mobiles _elements
     */
    bool detect(cv::Mat& diff, int thre);

    /**
     * @brief Builds motion mask from current frame. Simple difference between background and current frame.
     */
    void detect_simple(cv::Mat& current_frame);

    /**
     *@brief Builds motion mask from current frame. Similar as detect_simple but use a powerful adaptive background method.
     */
    void detect_MOG(cv::Mat& current_frame);

    /**
     *@brief Builds motion mask from depth frame. Uses the MOG background subtraction algorithm, with some tuning to work with depth frames. One may need to use RGBD utils to subscribe to both Depth and RBG topics on ROS.
     */
    void detect_MOG_depth(cv::Mat& depth_frame_16UC1);

    bool detect_on_cloud(const std::vector<double>& sv_center, PointCloudXYZ& diff_cloud, int threshold = 0,double dist_thres = 0.02, double mean_thres = 0.2, double octree_res = 0.02);

    /**
     * @brief setInputFrames
     * @param vector of 2 successives frames
     */
    void setInputFrames(const std::vector<cv::Mat>& f)
    { _frames = f; }

    void setInputClouds(const PointCloudT::Ptr& cloud1,const PointCloudT::Ptr& cloud2)
    {
        _cloud_frames.resize(2);

        _cloud_frames[0] = cloud1;
        _cloud_frames[1] = cloud2;
    }

    /**
     * @brief rect_clustering : Fonction for clustering the bound rectangle of detected object.
     * The object could be detected by part.
     * @param rect_array
     *
     */
    void rect_clustering(std::vector<cv::Rect>& rect_array);

    /**
     * @brief save results into little image all detected element.
     * @param counter
     */
    void save_results(const std::string& folder, int counter);

    /**
     * @brief Extracts ROIs as cv::Rect from a given motion mask.
     */
    std::vector<cv::Rect> motion_to_ROIs(cv::Mat& motion_mask, int thres = 75);

    /**
     * @brief get the results in images
     * @return vector of cv::Mat
     */
    std::vector<cv::Mat> getResults()
    { return _results; }

    /**
     * @brief get the results in rectangles form
     * @return vector cv::Rect
     */
    std::vector<cv::Rect> getResultsRects()
    { return _resultsRects; }


private :
    std::vector<cv::Mat> _frames;
    std::vector<PointCloudT::Ptr> _cloud_frames;
    std::vector<cv::Mat> _results;
    std::vector<cv::Rect> _resultsRects;

    std::array<int, 3> _kernels_size = {{1, 1, 1}};
    std::array<int, 3> _thresholds = {{25, 25, 25}};
    std::array<cv::Mat, 3> _elements;

    cv::Mat _background_BGR;
    cv::Mat _background_depth_16UC1;

    cv::BackgroundSubtractorMOG2 _background_sub_MOG2;

    //parameter
    double _minArea; //minimum size of contours;

    /**
     * @brief Fills a vector of matrices by selecting ROIs in current frame.
     */
    void extractResults(std::vector<cv::Rect>& rects);

    /**
     * @brief Used by detect_simple to progressively update the background frame by blending it with the current one.
     */
    void linear_background_blend(std::vector<cv::Mat>& background_channels,
                                 std::vector<cv::Mat>& current_frame_channels);

    /**
     * @brief Denoises depth frames by doing smoothing average over frames and using a closing morphological transformation.
     */
    void denoise_depth(cv::Mat& depth_frame_16UC1);
};
}

#endif //MOTION_DETECTION_H
