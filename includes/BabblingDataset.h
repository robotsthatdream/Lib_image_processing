#ifndef _BABBLING_DATASET_H
#define _BABBLING_DATASET_H

#include <pcl_types.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <MotionDetection.h>
#include <SurfaceOfInterest.h>
#include <pcl/filters/passthrough.h>

namespace image_processing{

class BabblingDataset{

public:


    typedef std::vector<PointCloudT> cloud_set_t;
    typedef std::map<double,cloud_set_t> cloud_trajectories_t;
    typedef std::map<int,cloud_trajectories_t> cloud_trajectories_set_t;
    typedef std::map<double,std::pair<cv::Mat,cv::Mat>> rgbd_set_t;
    typedef std::map<int,rgbd_set_t> per_iter_rgbd_set_t;
    typedef std::vector<cv::Rect> rect_set_t;
    typedef std::map<double,rect_set_t> rect_trajectories_t;
    typedef std::map<int,rect_trajectories_t> rect_trajectories_set_t;

    BabblingDataset(){}
    BabblingDataset(const BabblingDataset& bds)
        : _per_iter_rgbd_set(bds._per_iter_rgbd_set),
          _per_iter_rect_set(bds._per_iter_rect_set),
          _archive_name(bds._archive_name),
          _camera_parameter(bds._camera_parameter),
          _data_structure(bds._data_structure),
          _iterations_folders(bds._iterations_folders){}

    BabblingDataset(const std::string& arch_name){
        _load_iteration_folders(arch_name);
    }
    BabblingDataset(const std::string &arch_name, const std::string &meta_data){
        _archive_name = arch_name;
        _load_iteration_folders(arch_name);
        _load_data_structure(meta_data);
    }


    bool load_dataset(const std::string& meta_data_filename, const std::string &arch_name, int iteration = 0);
    bool load_dataset(int iteration = 0);
    void extract_cloud_trajectories(cloud_trajectories_set_t& cloud_traj);
    std::pair<double,cloud_set_t> extract_cloud(const rgbd_set_t::const_iterator &iter,
                                                const rect_trajectories_t::const_iterator &rect_iter);

    void _rgbd_to_pointcloud(const cv::Mat &rgb, const cv::Mat &depth, PointCloudT::Ptr ptcl);

    //GETTERS
    const per_iter_rgbd_set_t& get_per_iter_rgbd_set(){return _per_iter_rgbd_set;}
    const rect_trajectories_set_t& get_per_iter_rect_set(){return _per_iter_rect_set;}

private:
    per_iter_rgbd_set_t _per_iter_rgbd_set;
    rect_trajectories_set_t _per_iter_rect_set;
    YAML::Node _data_structure;
    std::map<int,std::string> _iterations_folders;
    YAML::Node _camera_parameter;
    YAML::Node _supervoxel_parameter;
    YAML::Node _soi_parameter;
    workspace_t _workspace_parameter;
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
    bool _load_hyperparameters(const std::string& filename);
    bool _load_rgbd_images(const std::string &foldername, const rect_trajectories_t& rects, rgbd_set_t& rgbd_set);
//    void _rgbd_to_pointcloud(const cv::Mat& rgb, const cv::Mat& depth, PointCloudT::Ptr ptcl);

};
}

#endif //_BABBLING_DATASET_H
