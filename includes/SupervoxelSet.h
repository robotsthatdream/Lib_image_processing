#ifndef SUPERVOXEL_SEGMENT_H
#define SUPERVOXEL_SEGMENT_H

#include <pcl/filters/filter_indices.h>
#include <pcl/segmentation/segment_differences.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/distances.h>

#include <pcl_types.h>
#include <string>
#include <vector>

#include <memory>


namespace image_processing {



/**
 * @brief The SupervoxelSet class
 * This class have several goal. It can compute the supervoxels clustering with a given pointcloud or be construct "piece by piece".
 * It could represent a segment of a scene with a supervoxel clustering as structure.
 */
class SupervoxelSet{

public :

    /**
     * @brief constant parameter for super voxel clustering
     */
    struct parameters{
        static constexpr bool use_transform = false;
        static constexpr float voxel_resolution = 0.008f;
        static constexpr float color_importance = 0.01f;
        static constexpr float spatial_importance = 0.4f;
        static constexpr float normal_importance = 0.4f;
        static constexpr float seed_resolution = 0.05f;
    };

    typedef std::shared_ptr<SupervoxelSet> Ptr;
    typedef const std::shared_ptr<SupervoxelSet> ConstPtr;

    SupervoxelSet(){
        _extractor.reset(new pcl::SupervoxelClustering<PointT>(parameters::voxel_resolution,parameters::seed_resolution,parameters::use_transform));
    }
    SupervoxelSet(const PointCloudT::Ptr& cloud) : _inputCloud(cloud){
        _extractor.reset(new pcl::SupervoxelClustering<PointT>(parameters::voxel_resolution,parameters::seed_resolution,parameters::use_transform));
    }
    SupervoxelSet(const SupervoxelSet& super) :
        _inputCloud(super._inputCloud),
        _supervoxels(super._supervoxels),
        _adjacency_map(super._adjacency_map),
        _extractor(_extractor){}

    //METHODES-------------------------------------------------
    /**
     * @brief compute the supervoxels with the input cloud
     * @param nbr_iteration (set nbr_iteration > 1 if you want to refine the supervoxels) default value = 1
     * @return colorized pointcloud. Each color correspond to a supervoxel for a vizualisation.
     */
    void computeSupervoxel(std::vector<float> area = std::vector<float>());

    /**
     * @brief extract a pointcloud of edges of each supervoxel
     * @param edges_cloud output pointcloud
     * @param supervoxel_adjacency (optional)
     */
    void extractEdges(PointCloudT::Ptr edges_cloud, AdjacencyMap supervoxel_adjacency = AdjacencyMap());

    /**
     *@brief insert a new supervoxel in this. (Whatever his neighborhood)
     *@param label : uint32_t
     *@param supervoxel : pcl::Supervoxel
     *@param neighborhood : std::vector<uint32_t> neighborLabel
     */
    void insert(uint32_t label ,pcl::Supervoxel<PointT>::Ptr supervoxel, std::vector<uint32_t> neighborLabel);

    /**
     *@brief Restore coherence between supervoxels and adjacency_map.
     * Erase all member in adjacency_map who doesn't belong to supervoxels
     */
    void consolidate();

    /**
     *@brief search a supervoxel who correspond to label in this
     *@param label : uint32_t
     *@return if this contain the supervoxel who correspond to label : bool
     */
    bool contain(uint32_t label);

    /**
     *@brief remove the supervoxel who correspond to label
     *@param label : uint32_t
     */
    void remove(uint32_t label);

    /**
     * @brief is this empty ?
     * @return true if this is empty false otherwise
     */
    bool empty(){return _supervoxels.empty() && _adjacency_map.empty();}

    /**
     * @brief clear
     */
    void clear();

    /**
     * @brief extractCloud : give the pointcloud in base of supervoxel
     * @param resultCloud : output
     */
    void extractCloud(PointCloudT &resultCloud);

    /**
     * @brief globalPosition
     * @return global position of this segment
     */
    pcl::PointXYZ globalPosition();

    /**
     * @brief search which supervoxel contain the position (x,y,z)
     * @param x
     * @param y
     * @param z
     * @return label of voxel, if the position is in any voxel the value will be 221133.
     */
    uint32_t whichVoxelContain(float x, float y, float z);

    /**
     * @brief comparison function between this and a given SupervoxelSet
     * @param a set of supervoxel
     * @param threshold
     * @param position_importance
     * @param normal_importance
     * @return the SupervoxelSet of difference between this and super
     */
    SupervoxelSet compare(SupervoxelSet &super, double threshold, double position_importance = 1, double normal_importance = 1);


    /**
     * @brief substract
     * @param cloud
     */
    void substract(SupervoxelSet &cloud);
    //---------------------------------------------------------

    //SETTERS & GETTERS----------------------------------------
    /**
     * @brief extract the pointcloud of center of each supervoxel
     * @param centroids : output pointcloud
     * @param centroidsLabel : output labels to retrieve the correspondance between the supervoxel and his center.
     * @param centroid_normals : output pointcloud of centroids normals (in second variant)
     */
    void getCentroidCloud(PointCloudT& centroids, std::map<int,uint32_t>& centroidsLabel, PointCloudN& centroid_normals);
    void getCentroidCloud(PointCloudT& centroids, std::map<int,uint32_t>& centroidsLabel);

    /**
     * @brief getColoredCloud
     * @return a colored cloud to visualize supervoxel clustering
     */
    const PointCloudT& getColoredCloud(){
        return *(_extractor->getColoredCloud());
    }

    /**
     * @brief setInputCloud
     * @param cloud
     */
    void setInputCloud(const PointCloudT::Ptr& cloud){_inputCloud = cloud;}

    /**
     * @brief getInputCloud
     * @return
     */
    const PointCloudT::Ptr& getInputCloud(){return _inputCloud;}

    /**
     * @brief getAdjacencyMap
     * @return
     */
    AdjacencyMap getAdjacencyMap(){return _adjacency_map;}

    /**
     * @brief getSupervoxels
     * @return
     */
    SupervoxelArray getSupervoxels(){return _supervoxels;}

    /**
     * @brief setSeedResolution
     * @param sr
     */
    void setSeedResolution(float sr){_extractor->setSeedResolution(sr);}

    /**
     *@brief compute the neighborhood (first layer) of a given supervoxel
     *@param label : uint32_t
     *@return neighborhood : std::vector<uint32_t>
     */
    std::vector<uint32_t> getNeighbor(uint32_t label);

    /**
     *@brief acces methodes to a supervoxel
     *@param label : uint32_t
     *@return pcl::Supervoxel
     */
    const pcl::Supervoxel<PointT>::Ptr& at(uint32_t label){return _supervoxels.at(label);}
    //---------------------------------------------------------

protected:
    uint32_t isInThisVoxel(float x, float y, float z, uint32_t label, AdjacencyMap am, boost::random::mt19937 gen, int counter = 5);

    PointCloudT::Ptr _inputCloud;
    std::shared_ptr<pcl::SupervoxelClustering<PointT> > _extractor;
    SupervoxelArray _supervoxels;
    AdjacencyMap _adjacency_map;
    double _seed_resolution;

};

}//image_processing
#endif //SUPERVOXEL_SEGMENT_H
