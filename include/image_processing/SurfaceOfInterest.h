#ifndef _SURFACE_OF_INTEREST_H
#define _SURFACE_OF_INTEREST_H

#include "SupervoxelSet.h"
//#include "TrainingData.hpp"
#include <boost/random.hpp>
#include <ctime>



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

    typedef std::map<uint32_t,double> saliency_map_t;

    /**
     * @brief default constructor
     */
    SurfaceOfInterest() : SupervoxelSet(){
        _gen.seed(std::time(0));
        _weights.emplace("color",saliency_map_t());
        _weights.emplace("normal",saliency_map_t());
    }

    /**
     * @brief constructor with a given input cloud
     * @param input cloud
     */
    SurfaceOfInterest(const PointCloudT::Ptr& cloud) : SupervoxelSet(cloud){
        _gen.seed(std::time(0));
        _weights.emplace("color",saliency_map_t());
        _weights.emplace("normal",saliency_map_t());
    }
    /**
     * @brief copy constructor
     * @param soi
     */
    SurfaceOfInterest(const SurfaceOfInterest& soi) :
        SupervoxelSet(soi),
        _weights(soi._weights),
        _labels(soi._labels),
        _labels_no_soi(_labels_no_soi)
    {
        _gen.seed(std::time(0));
        _weights.emplace("color",saliency_map_t());
        _weights.emplace("normal",saliency_map_t());
    }
    /**
     * @brief constructor with a SupervoxelSet
     * @param super
     */
    SurfaceOfInterest(const SupervoxelSet& super) : SupervoxelSet(super){
        _gen.seed(std::time(0));
    }

    /**
     * @brief methode to find soi, with a list of keypoints this methode search in which supervoxel are this keypoints
     * @param key_pts
     */
    void find_soi(const PointCloudXYZ::Ptr key_pts);

    /**
     * @brief NAIVE POLICY generate the soi for a pure random choice (i.e. all supervoxels are soi)
     * @param workspace
     * @return if the generation of soi is successful
     */
    bool generate(workspace_t& workspace);

    /**
     * @brief LEARNING POLICY generate the soi with a classifier specify in argument
     * @param the classifier
     * @param workspace
     * @return if the generation of soi is successful
     */
    template <typename classifier_t>
    bool generate(const std::string &modality, classifier_t& classifier, workspace_t& workspace, float init_val = 1.){
        if(!computeSupervoxel(workspace))
        return false;

        init_weights(init_val);
        compute_weights(modality, classifier);
        return true;
    }

    /**
     * @brief KEY POINTS POLICY generate the soi with key points. Soi will be supervoxels who contains at least one key points.
     * @param key points
     * @param workspace
     * @return if the generation of soi is successful
     */
    bool generate(const PointCloudXYZ::Ptr key_pts, workspace_t &workspace);

    /**
     * @brief EXPERT POLICY generate the soi by deleting the background
     * @param pointcloud of the background
     * @param workspace
     * @return if the generation of soi is successful
     */
    bool generate(const PointCloudT::Ptr background, workspace_t& workspace);

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
    template <typename classifier_t>
    void compute_weights(const std::string& modality, classifier_t &classifier){
        //For Color case
        if(modality == "color"){
            for(const auto& sv : _supervoxels){
                float hsv[3];
                tools::rgb2hsv(sv.second->centroid_.r,
                               sv.second->centroid_.g,
                               sv.second->centroid_.b,
                               hsv[0],hsv[1],hsv[2]);
                Eigen::VectorXd new_s(3);
                new_s << hsv[0],
                        hsv[1],
                        hsv[2];
                _weights["color"][sv.first] = classifier.compute_estimation(new_s,1);
            }
            return;
        }
        if(modality == "normal"){
            for(const auto& sv : _supervoxels){
                Eigen::VectorXd new_s(3);
                new_s << sv.second->normal_.normal[0],
                        sv.second->normal_.normal[1],
                        sv.second->normal_.normal[2];
                _weights["normal"][sv.first] = classifier.compute_estimation(new_s,1);
            }
            return;
        }

        std::cerr << "SurfaceOfInterest Error: unknow modality" << std::endl;
    }


    /**
     * @brief choose randomly one soi
     * @param supervoxel
     * @param label of chosen supervoxel in the soi set
     */
    bool choice_of_soi(const std::string &modality, pcl::Supervoxel<PointT> &supervoxel, uint32_t& lbl);
    bool choice_of_soi_by_uncertainty(const std::string &modality, pcl::Supervoxel<PointT> &supervoxel, uint32_t &lbl);

    /**
     * @brief delete the background of the input cloud
     * @param a pointcloud
     */
    void delete_background(const PointCloudT::Ptr background);


    PointCloudT getColoredWeightedCloud(const std::string &modality);


private :
    std::vector<uint32_t> _labels;
    std::vector<uint32_t> _labels_no_soi;
    std::map<std::string,saliency_map_t> _weights;


    boost::random::mt19937 _gen;

};

}

#endif //_SURFACE_OF_INTEREST_H
