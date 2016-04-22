#ifndef _SURFACE_OF_INTEREST_H
#define _SURFACE_OF_INTEREST_H

#include "SupervoxelSet.h"
#include "TrainingData.hpp"
#include <boost/random.hpp>
#include <ctime>

namespace image_processing {

class SurfaceOfInterest : public SupervoxelSet
{
public:
    /**
     * @brief default constructor
     */
    SurfaceOfInterest() : SupervoxelSet(){
        _gen.seed(std::time(0));
        init_soi<parameters::soi>();
    }

    /**
     * @brief constructor with a given input cloud
     * @param input cloud
     */
    SurfaceOfInterest(const PointCloudT::Ptr& cloud) : SupervoxelSet(cloud){
        _gen.seed(std::time(0));
        init_soi<parameters::soi>();
    }
    /**
     * @brief copy constructor
     * @param soi
     */
    SurfaceOfInterest(const SurfaceOfInterest& soi) : SupervoxelSet(soi){
        _gen.seed(std::time(0));
        init_soi<parameters::soi>();
    }
    /**
     * @brief constructor with a SupervoxelSet
     * @param super
     */
    SurfaceOfInterest(const SupervoxelSet& super) : SupervoxelSet(super){
        _gen.seed(std::time(0));
        init_soi<parameters::soi>();
    }

    /**
     * @brief initialisation function
     * Template : parameters use for the soi comuptation
     */
    template <typename Param>
    void init_soi(){
        _param.color_normal_ratio = Param::color_normal_ratio;
        _param.distance_threshold = Param::distance_threshold;
        _param.interest_increment = Param::interest_increment;
        _param.non_interest_val = Param::non_interest_val;
//        _param.normal_importance = Param::normal_importance;
    }

    /**
     * @brief methode to find soi, with a list of keypoints this methode search in which supervoxel are this keypoints
     * @param key_pts
     */
    void find_soi(const PointCloudXYZ::Ptr key_pts);

    /**
     * @brief reduce the set of supervoxels to set of soi
     */
    void reduce_to_soi();

    void init_weights();

    /**
     * @brief methode compute the weight of each supervoxels. The weights represent the probability for a soi to be explored.
     * All weights are between 0 and 1.
     * @param lbl label of the explored supervoxel
     * @param interest true if the explored supervoxel is interesting false otherwise
     */
    void compute_weights(const TrainingData<pcl::Supervoxel<PointT>>& data);

    /**
     * @brief choose randomly one soi
     * @param supervoxel
     * @param label of chosen supervoxel in the soi set
     */
    void choice_of_soi(pcl::Supervoxel<PointT>& supervoxel,uint32_t& lbl);

    PointCloudT getColoredWeightedCloud();

private :

    void _compute_distances(std::map<uint32_t,float> &distances, pcl::Supervoxel<PointT> sv);

    float _L2_distance(const std::vector<float> &p1,const std::vector<float> &p2);

    struct _param_t{
        float interest_increment;
        float non_interest_val;
        float color_normal_ratio;
        float distance_threshold;
    };

    _param_t _param;

    std::vector<uint32_t> _labels;
    std::vector<uint32_t> _labels_no_soi;
    std::map<uint32_t,float> _weights;
    boost::random::mt19937 _gen;

    std::vector<std::pair<bool,float>> _data;

};

}

#endif //_SURFACE_OF_INTEREST_H
