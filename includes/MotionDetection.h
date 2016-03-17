#ifndef MOTION_DETECTION_H
#define MOTION_DETECTION_H

#include <opencv2/opencv.hpp>

/**
 * @brief The MotionDetection class
 * This class provide tools for motion detection based on rgb video camera.
 *
 */
class MotionDetection{

public:
    /**
     * @brief MotionDetection
     * default constructor
     */
    MotionDetection();

    /**
     * @brief MotionDetection
     * @param vector of 2 successives frames
     * @param minimum size of detection
     */
    MotionDetection(std::vector<cv::Mat> f, double min_area = 100) : _frames(f), minArea(min_area) {}

    /**
     * @brief detect
     * @param image with the contours of mobiles elements
     */
    void detect(cv::Mat &diff);

    /**
     * @brief setInputFrames
     * @param vector of 2 successives frames
     */
    void setInputFrames(std::vector<cv::Mat> f){_frames = f;}

    /**
     * @brief rect_clustering : Fonction for clustering the bound rectangle of detected object.
     * The object could be detected by part.
     * @param rect_array
     *
     */
    void rect_clustering(std::vector<cv::Rect> & rect_array);

    /**
     * @brief save results into little image all detected element.
     * @param counter
     */
    void save_results(const std::string &folder, int counter);

    /**
     * @brief get the results in images
     * @return vector of cv::Mat
     */
    std::vector<cv::Mat> getResults(){return _results;}

    /**
     * @brief get the results in rectangles form
     * @return vector cv::Rect
     */
    std::vector<cv::Rect> getResultsRects(){return _resultsRects;}


private :
    std::vector<cv::Mat> _frames;
    std::vector<cv::Mat> _results;
    std::vector<cv::Rect> _resultsRects;

    //parameter
    double minArea; //minimum size of contours;

    void extractResults(std::vector<cv::Rect> &rects);


};

#endif //MOTION_DETECTION_H
