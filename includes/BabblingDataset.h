#ifndef _BABBLING_DATASET_H
#define _BABBLING_DATASET_H

#include <pcl_types.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

namespace image_processing{

class BabblingDataset{

public:
    typedef std::map<double,PointCloudT> cloud_trajectory_t;
    typedef std::map<int,cloud_trajectory_t> cloud_trajectory_set_t;
    typedef std::map<double,std::pair<cv::Mat,cv::Mat>> rgbd_set_t;
    typedef std::map<int,rgbd_set_t> per_iter_rgbd_set_t;
    typedef std::vector<cv::Rect> rect_set_t;
    typedef std::map<double,rect_set_t> rect_trajectories_t;
    typedef std::map<int,rect_trajectories_t> rect_trajectories_set_t;

    BabblingDataset(){}
    BabblingDataset(const BabblingDataset& bds)
        : _per_iter_rgbd_set(bds._per_iter_rgbd_set),
          _per_iter_rect_set(bds._per_iter_rect_set){}

    BabblingDataset(const std::string& arch_name){
        _load_iteration_folders(arch_name);
    }
    BabblingDataset(const std::string &arch_name, const std::string &meta_data){
        _load_iteration_folders(arch_name);
        _load_data_structure(meta_data);
        _archive_name = arch_name;
    }


    bool load_dataset(const std::string& meta_data_filename, const std::string &arch_name, int iteration = 0);
    bool load_dataset(int iteration = 0);



    //GETTERS
    const per_iter_rgbd_set_t& get_per_iter_rgbd_set(){return _per_iter_rgbd_set;}
    const rect_trajectories_set_t& get_per_iter_rect_set(){return _per_iter_rect_set;}
    void _rgbd_to_pointcloud(const cv::Mat& rgb, const cv::Mat& depth, PointCloudT::Ptr ptcl);
    bool _load_camera_param(const std::string& filename);

private:
    per_iter_rgbd_set_t _per_iter_rgbd_set;
    rect_trajectories_set_t _per_iter_rect_set;
    YAML::Node _data_structure;
    std::map<int,std::string> _iterations_folders;
    YAML::Node _camera_parameter;
    std::string _archive_name;

    /**
     * @brief load all iteration folder name
     * @param archive name
     */
    void _load_iteration_folders(const std::string& arch_name);

    /**
     * @brief load the meta data about the data structure
     * @param meta data filename
     * @return if success
     */
    bool _load_data_structure(const std::string& meta_data_filename);
    bool _load_data_iteration(const std::string& foldername, rgbd_set_t& rgbd_set, rect_trajectories_t& rect_traj);
    bool _load_motion_rects(const std::string& filename, rect_trajectories_t &rect_traj);
//    bool _load_camera_param(const std::string& filename);
    bool _load_rgbd_images(const std::string &foldername, const rect_trajectories_t& rects, rgbd_set_t& rgbd_set);

//    void _rgbd_to_pointcloud(const cv::Mat& rgb, const cv::Mat& depth, PointCloudT& ptcl);

};
}

#endif //_BABBLING_DATASET_H
