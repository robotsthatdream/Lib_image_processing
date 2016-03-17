#include <MotionDetection.h>
#include <fstream>
#include <ctime>

void MotionDetection::detect(cv::Mat& diff){

    if(_frames.size() != 2){
        std::cerr << "detect : need exactly 2 frames" << std::endl;
        return;
    }

    cv::Mat current = _frames[1].clone();
    cv::Mat previous = _frames[0].clone();

    diff = cv::Mat::zeros(current.rows,current.cols,current.type());

    //conversion color to grayscale
    cv::cvtColor(previous,previous,cv::COLOR_BGR2GRAY);
    cv::cvtColor(current,current,cv::COLOR_BGR2GRAY);

    //gaussian blur to eliminate some noise
    cv::GaussianBlur(previous,previous,cv::Size(3,3),0);
    cv::GaussianBlur(current,current,cv::Size(3,3),0);

    //compute the difference between the two frame to detect a motion
    cv::subtract(current,previous,diff);

    //binarisation of the difference image
    cv::adaptiveThreshold(diff,diff,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY_INV,5,2);

    //search for contours in difference image to detect the different mobile object
    std::vector<cv::Mat> contours;
    cv::findContours(diff,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);

    //selection of minimal size contours using bounding box;

    std::vector<cv::Rect> rects;
    for(int i = 0; i < contours.size(); i++){
        double area = cv::contourArea(contours[i]);
        if(area > minArea){
//            cv::drawContours(current,contours,i,cv::Scalar(0,0,255));
            rects.push_back(cv::boundingRect(contours[i]));
//            cv::rectangle(current,rects.back(),cv::Scalar(255,0,0));

        }
    }

//    cv::imshow("current",_frames[1]);
//    cv::imshow("previous",_frames[0]);
//    cv::waitKey(30);

    //clustering of the bounding boxes to assemble the parted objects
    rect_clustering(rects);
    _resultsRects = rects;
    extractResults(rects);
}

void MotionDetection::save_results(const std::string& folder,int counter){
  if(!_resultsRects.empty())
    {
      std::stringstream stream;
      stream << folder << "image_rects_info_" << counter << ".txt";
      std::ofstream file(stream.str().c_str(),std::ios::out | std::ios::app);
      if(file){
          std::time_t currentTime = std::time(NULL);
          std::string currentDate(std::ctime(&currentTime));
          file << currentDate << "\n";
        }

      for(int i = 0; i < _results.size(); i++){
          std::stringstream ss;
          ss << folder << "seg_" << counter << "_" << i <<  ".png";

          if(!cv::imwrite(ss.str(),_results[i])){
              std::cerr << "error cv::imwrite for file "  << ss.str() <<  std::endl;
              return;
            }
          if(file){

              cv::Rect rect = _resultsRects[i];
              file << "seg_" << counter << "_" << i << " pose : (" << rect.x << ";" << rect.y << ") size : (" << rect.height << ";" << rect.width << ")\n";
            }
        }
      if(file)
        file.close();
    }
}

void MotionDetection::rect_clustering(std::vector<cv::Rect> &rect_array){
    if(rect_array.size() > 1){
        bool b = false;
        for(int i = 0; i < rect_array.size(); i++){
            cv::Rect tmp = rect_array[i];
            for(int j = 0 ; j < rect_array.size() ; j++){
                cv::Rect tmp2 = rect_array[j];
                if(j != i && ((tmp & tmp2).area() != 0)){
                    tmp |= tmp2;
                    rect_array.erase(rect_array.begin() + j);
                    b = true;
                }
            }
            if(b){
                rect_array.push_back(tmp);
                rect_array.erase(rect_array.begin()+i);
                i--;
                b = false;
            }
        }

    }
}

void MotionDetection::extractResults(std::vector<cv::Rect>& rects){
    for(int i = 0; i < rects.size(); i++)
        _results.push_back(cv::Mat(_frames[1],rects[i]));
}
