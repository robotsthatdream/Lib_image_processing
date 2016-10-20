#ifndef _SURFACE_OF_INTEREST_H
#define _SURFACE_OF_INTEREST_H

#include "SupervoxelSet.h"
#include "TrainingData.hpp"
#include <boost/random.hpp>
#include <ctime>

#include <online-machine-learning/online_rf.h>

namespace image_processing {

typedef struct SvFeature{

    SvFeature(){}
    SvFeature(std::vector<double> c, std::vector<double> n) :
        color(c), normal(n){}

    std::vector<double> color;
    std::vector<double> normal;
}SvFeature;

inline std::ostream& operator<<(std::ostream& os, const SvFeature& feature){

    os << "color : ";
    for(auto c : feature.color)
        os << c << ";";

    os << "normal : ";
    for(auto n : feature.normal)
        os << n << ";";


    return os;
}

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
     * @brief generate the soi for a pure random choice (i.e. all supervoxels are soi)
     * @param workspace
     * @return if the generation of soi is successful
     */
    bool generate(workspace_t& workspace);

    /**
     * @brief generate the soi with a simple linear fuzzy classifier
     * @param training dataset
     * @param workspace
     * @return if the generation of soi is successful
     */
    bool generate(const TrainingData<SvFeature> &dataset, workspace_t& workspace);

    /**
     * @brief generate the soi with key points. Soi will be supervoxels who contains at least one key points.
     * @param key points
     * @param workspace
     * @return if the generation of soi is successful
     */
    bool generate(const PointCloudXYZ::Ptr key_pts, workspace_t &workspace);

    /**
     * @brief generate the soi by deleting the background
     * @param pointcloud of the background
     * @param workspace
     * @return if the generation of soi is successful
     */
    bool generate(const PointCloudT::Ptr background, workspace_t& workspace);

    /**
     * @brief generate the soi by classifing the supervoxel with a online trained Random Forest.
     * If the generation is in training mode, the soi will be computed according to an interest to unknown area in the visual field.
     * In this mode the classifier should be in training. In the non training mode, the classifier is fixed and the soi are computed according to the classification.
     * @param RF classifier
     * @param workspace
     * @param if true, the soi will be generated for training mode.
     * @return if the generation of soi is successful
     */
    bool generate(const std::shared_ptr<oml::Classifier> model, workspace_t &workspace, bool training = true);

    /**
     * @brief reduce the set of supervoxels to set of soi
     */
    void reduce_to_soi();

    void init_weights(float value = 1.);

    /**
     * @brief methode compute the weight of each supervoxels. The weights represent the probability for a soi to be explored.
     * All weights are between 0 and 1.
     * @param lbl label of the explored supervoxel
     * @param interest true if the explored supervoxel is interesting false otherwise
     */
    void compute_weights(const TrainingData<SvFeature> &data);

    /**
     * @brief compute the weights of each supervoxel with an online trained Random Forest classifer
     * @param model
     */
    void compute_weights(const std::shared_ptr<oml::Classifier> model);
    void compute_confidence_weights(const std::shared_ptr<oml::Classifier> model);

    /**
     * @brief choose randomly one soi
     * @param supervoxel
     * @param label of chosen supervoxel in the soi set
     */
    void choice_of_soi(pcl::Supervoxel<PointT> &supervoxel, uint32_t& lbl);

    /**
     * @brief delete the background of the input cloud
     * @param a pointcloud
     */
    void delete_background(const PointCloudT::Ptr background);


    PointCloudT getColoredWeightedCloud();


private :

    void _compute_distances(std::map<uint32_t,float> &distances, const SvFeature &sv);

    double _L2_distance(const std::vector<double> &p1, const std::vector<double> &p2);

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
