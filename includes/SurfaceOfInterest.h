#ifndef _SURFACE_OF_INTEREST_H
#define _SURFACE_OF_INTEREST_H

#include <SupervoxelSet.h>
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
    }

    /**
     * @brief constructor with a given input cloud
     * @param input cloud
     */
    SurfaceOfInterest(const PointCloudT::Ptr& cloud) : SupervoxelSet(cloud){
        _gen.seed(std::time(0));
    }
    /**
     * @brief copy constructor
     * @param soi
     */
    SurfaceOfInterest(const SurfaceOfInterest& soi) : SupervoxelSet(soi){
        _gen.seed(std::time(0));
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
     * @brief reduce the set of supervoxels to set of soi
     */
    void reduce_to_soi();

    /**
     * @brief methode compute the weight of each supervoxels. The weights represent the probability for a soi to be explored.
     * All weights are between 0 and 1.
     * TODO
     */
    void compute_weights();

    /**
     * @brief choose randomly one soi
     * @param supervoxel
     * @param label of chosen supervoxel in the soi set
     */
    void choice_of_soi(pcl::Supervoxel<PointT>& supervoxel,uint32_t& lbl);

private :
    std::vector<uint32_t> _labels;
    std::map<uint32_t,float> _weights;
    boost::random::mt19937 _gen;

};

}

#endif //_SURFACE_OF_INTEREST_H
