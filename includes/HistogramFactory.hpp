#ifndef HISTOGRAM_FACTORY_HPP
#define HISTOGRAM_FACTORY_HPP

#include <map>
#include <vector>
#include <pcl_types.h>
#include <memory>
#include <Eigen/Core>

#include "SupervoxelSet.h"

#include DIM 6

namespace image_processing {

class HistogramFactory {

public:
    /**
     * @brief default constructor
     */
    HistogramFactory(){
        _histograms.resize(DIM);
        _size = 0;
        _with_difference = false;
    }

    /**
     * @brief constructor for 6 dimensional features sample. This constructor will compute a sample with (r,g,b,n1,n2,n3) of sv
     * @param sv
     */
    HistogramFactory(const pcl::Supervoxel<PointT>& sv){
        _histograms.resize(DIM);
        _with_difference = false;
        compute(sv);
    }

    /**
     * @brief Constructor for features with neighborhood. This constructor will compute a sample with features of 6*N (N size of neighborhood)
     * Caution the samples will have different nbr of feature
     * @param sv
     * @param neighbors
     * @param if diff
     */
    HistogramFactory(const pcl::Supervoxel<PointT>& sv, const std::vector<pcl::Supervoxel<PointT>>& neighbors, bool diff = false){
        _histograms.resize(DIM);
        _with_difference = diff;
        compute(sv,neighbors);
    }
    /**
     * @brief copy constructor
     * @param HF
     */
    HistogramFactory(const HistogramFactory& HF) :
        _size(HF._size), _histograms(HF._histograms)
    {}

    /**
     * @brief compute the histograms related to sv
     * @param sv
     * @param (optional) neighbors of sv
     */
    void compute(const pcl::Supervoxel<PointT>& sv,
                 const std::vector<pcl::Supervoxel<PointT>>& neighbors = std::vector<pcl::Supervoxel<PointT>>());


    //GETTERS & SETTERS

   void set_with_difference(bool d){
        _with_difference = d;
   }

    /**
     * @brief get_histogram
     * @return the whole histogram
     */
    Eigen::VectorXd get_histogram(){
        Eigen::VectorXd res(_size*_histograms.size());
        res << _histograms[0], _histograms[1], _histograms[2], _histograms[3];
        return res;
    }
    /**
     * @brief return red histogram if i = 0, green histogram if i = 1, blue histogram if i = 2, depth histogram if i = 3,
     * @param i
     * @return histogram
     */
    Eigen::VectorXd get_histogram(int i){
        return _histograms[i];
    }


private:
    size_t _size;
    std::vector<Eigen::VectorXd> _histograms;
    bool _with_difference;

};

}

#endif //HISTOGRAM_FACTORY_HPP
