#include "image_processing/MotionDetection.h"
#include <pcl/registration/icp.h>

using namespace image_processing;

bool MotionDetection::detect(cv::Mat& diff, int thre)
{
    if (_frames.size() != 2) {
        std::cerr << "detect : need exactly 2 frames" << std::endl;
        return false;
    }

    cv::Mat current = _frames[1].clone();
    cv::Mat previous = _frames[0].clone();

    diff = cv::Mat::zeros(current.rows, current.cols, current.type());

    //conversion color to grayscale
    if(previous.channels() == 3)
        cv::cvtColor(previous, previous, cv::COLOR_BGR2GRAY);
    if(current.channels() == 3)
        cv::cvtColor(current, current, cv::COLOR_BGR2GRAY);

    //gaussian blur to eliminate some noise
    cv::GaussianBlur(previous, previous, cv::Size(3, 3), 0);
    cv::GaussianBlur(current, current, cv::Size(3, 3), 0);


    //compute the difference between the two frame to detect a motion
    cv::subtract(current, previous, diff);


    //binarisation of the difference image
    cv::adaptiveThreshold(diff, diff, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 5, 2);


    _resultsRects = motion_to_ROIs(diff,thre);

    //clustering of the bounding boxes to assemble the parted objects

//    rect_clustering(_resultsRects);
    extractResults(_resultsRects);
    return !_resultsRects.empty();
}

void MotionDetection::detect_simple(cv::Mat& current_frame_BGR)
{
    std::vector<cv::Mat> cf_HSV_channels;
    std::vector<cv::Mat> bg_HSV_channels;
    std::vector<cv::Mat> motion_mask_channels;

    cv::Mat current_frame_HSV;
    cv::Mat background_HSV;

    cv::Mat motion_mask;
    cv::Mat motion_mask_S_and_V;

    if (_background_BGR.empty()) {
        current_frame_BGR.copyTo(_background_BGR);
        cv::cvtColor(_background_BGR, background_HSV, CV_BGR2HSV);
    }
    else {
        cv::cvtColor(current_frame_BGR, current_frame_HSV, CV_BGR2HSV);

        motion_mask.create(current_frame_HSV.rows, current_frame_HSV.cols, CV_8UC3);

        cv::split(background_HSV, bg_HSV_channels);
        cv::split(current_frame_HSV, cf_HSV_channels);
        cv::split(motion_mask, motion_mask_channels);

        for (short i = 1; i < 3; i++) {
            _elements[i] = cv::getStructuringElement(cv::MORPH_RECT, cv::Size_<int>(2 * _kernels_size[i] + 1,
                                                                                    2 * _kernels_size[i] + 1));

            cv::absdiff(bg_HSV_channels[i], cf_HSV_channels[i], motion_mask_channels[i]);

            cv::threshold(motion_mask_channels[i], motion_mask_channels[i], _thresholds[i], 255,
                          CV_THRESH_BINARY);

            cv::morphologyEx(motion_mask_channels[i], motion_mask_channels[i], cv::MORPH_OPEN, _elements[i]);
        }

        cv::bitwise_and(motion_mask_channels[1], motion_mask_channels[2], motion_mask_S_and_V);

        cv::morphologyEx(motion_mask_S_and_V, motion_mask_S_and_V, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size_<int>(15, 15)),
                         cv::Point_<int>(-1, -1), 3);

        _resultsRects = motion_to_ROIs(motion_mask_S_and_V);
        linear_background_blend(bg_HSV_channels, cf_HSV_channels);

        cv::merge(bg_HSV_channels, background_HSV);
    }
}

void MotionDetection::detect_MOG(cv::Mat& current_frame_BGR)
{
    cv::Mat motion_mask;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size_<int>(3, 3));

    _background_sub_MOG2.operator()(current_frame_BGR, motion_mask);
    _background_sub_MOG2.getBackgroundImage(_background_BGR);

    cv::morphologyEx(motion_mask, motion_mask, cv::MORPH_OPEN, element);

    _resultsRects = motion_to_ROIs(motion_mask);
}

void MotionDetection::detect_MOG_depth(cv::Mat& depth_frame_16UC1)
{
    cv::Mat motion_mask;

    denoise_depth(depth_frame_16UC1);

    _background_sub_MOG2.operator()(_background_depth_16UC1, motion_mask);

    cv::threshold(motion_mask, motion_mask, 0, 255, CV_THRESH_BINARY);

    _resultsRects = motion_to_ROIs(motion_mask);
}

bool MotionDetection::detect_on_cloud(const PointCloudXYZ::Ptr sv, const std::vector<double>& sv_center, PointCloudXYZ::Ptr diff_cloud ,
                                      int threshold, double dist_thres, double mean_thres, double octree_res){
    std::vector<int> index;
    double dist, min_dist, mean_dist = 0;

    pcl::octree::OctreePointCloudChangeDetector<PointT> octree(octree_res);

    octree.setInputCloud(_cloud_frames[1]);
    octree.addPointsFromInputCloud();

    octree.switchBuffers();

    octree.setInputCloud(_cloud_frames[0]);
    octree.addPointsFromInputCloud();

    octree.getPointIndicesFromNewVoxels(index);



    if(index.size() <= threshold){
        std::cout << "no difference !" << std::endl;
        return false;
    }

    for(int i : index)
        diff_cloud->push_back(pcl::PointXYZ(_cloud_frames[0]->points[i].x,
                                           _cloud_frames[0]->points[i].y,
                                           _cloud_frames[0]->points[i].z));


    pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
    icp.setMaxCorrespondenceDistance(0.05);
    icp.setInputSource(sv);
    icp.setInputTarget(diff_cloud);
    PointCloudXYZ output;
    icp.align(output);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;

    std::function<double(double,double,double,double,double,double)> distance =
            [](double x1, double x2, double y1, double y2, double z1, double z2) -> double {
        return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
    };

    min_dist = distance(_cloud_frames[0]->points[index[0]].x,sv_center[0],
                    _cloud_frames[0]->points[index[0]].y,sv_center[1],
                    _cloud_frames[0]->points[index[0]].z,sv_center[2]);

    for(int i : index){
        dist = distance(_cloud_frames[0]->points[i].x,sv_center[0],
                        _cloud_frames[0]->points[i].y,sv_center[1],
                        _cloud_frames[0]->points[i].z,sv_center[2]);
        mean_dist += dist;
        if(min_dist > dist)
            min_dist = dist;
    }

    mean_dist = mean_dist/((double)index.size());

    std::cout << "min distance is " << min_dist << " and mean distance is " << mean_dist <<  std::endl;

    if(mean_dist - mean_thres > -0.01)
        return false;

    if(dist_thres - min_dist > -0.01 )
        return true;

    return false;
}

void MotionDetection::save_results(const std::string& folder, int counter)
{
    if (!_resultsRects.empty()) {
        std::stringstream stream;
        stream << folder << "image_rects_info_" << counter << ".txt";
        std::ofstream file(stream.str().c_str(), std::ios::out | std::ios::app);
        if (file) {
            std::time_t currentTime = std::time(NULL);
            std::string currentDate(std::ctime(&currentTime));
            file << currentDate << "\n";
        }

        for (int i = 0; i < _results.size(); i++) {
            std::stringstream ss;
            ss << folder << "seg_" << counter << "_" << i << ".png";

            if (!cv::imwrite(ss.str(), _results[i])) {
                std::cerr << "error cv::imwrite for file " << ss.str() << std::endl;
                return;
            }
            if (file) {

                cv::Rect rect = _resultsRects[i];
                file << "seg_" << counter << "_" << i << " pose : (" << rect.x << ";" << rect.y << ") size : (" <<
                rect.height << ";" << rect.width << ")\n";
            }
        }
        if (file) {
            file.close();
        }
    }
}

void MotionDetection::rect_clustering(std::vector<cv::Rect>& rect_array)
{
    if (rect_array.size() > 1) {
        bool b = false;
        for (int i = 0; i < rect_array.size(); i++) {
            cv::Rect tmp = rect_array[i];
            for (int j = 0; j < rect_array.size(); j++) {
                cv::Rect tmp2 = rect_array[j];
                if (j != i && ((tmp & tmp2).area() != 0)) {
                    tmp |= tmp2;
                    rect_array.erase(rect_array.begin() + j);
                    b = true;
                }
            }
            if (b) {
                rect_array.push_back(tmp);
                rect_array.erase(rect_array.begin() + i);
                i--;
                b = false;
            }
        }

    }
}

void MotionDetection::extractResults(std::vector<cv::Rect>& rects)
{
    for (int i = 0; i < rects.size(); i++) {
        _results.push_back(cv::Mat(_frames[1], rects[i]));
    }
}

std::vector<cv::Rect> MotionDetection::motion_to_ROIs(cv::Mat& motion_mask,int thres)
{
    std::vector<cv::Mat> contours;
    std::vector<cv::Rect> ROIs;

    //search for contours in difference image to detect the different mobile object
    cv::findContours(motion_mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    //selection of minimal size contours using bounding box;
    for (unsigned int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > thres) {
            ROIs.push_back(cv::boundingRect(contours[i]));
        }
    }
    return ROIs;
}

void MotionDetection::linear_background_blend(std::vector<cv::Mat>& background_channels,
                                              std::vector<cv::Mat>& current_frame_channels)
{
    double alpha = 0.75;
    double beta = 1.0 - alpha;

    for (unsigned int i = 0; i < background_channels.size(); i++) {
        cv::addWeighted(background_channels[i], alpha, current_frame_channels[i], beta, 0, background_channels[0]);
    }
}

void MotionDetection::denoise_depth(cv::Mat& frame)
{
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                cv::Size_<int>(13, 13));
    cv::morphologyEx(frame, frame, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);

    if (_background_depth_16UC1.empty()) {
        frame.copyTo(_background_depth_16UC1);
    }
    else {
        cv::addWeighted(_background_depth_16UC1, 0.70, frame, 0.30, 0.0, _background_depth_16UC1);
    }

}



