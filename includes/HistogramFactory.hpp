#ifndef HISTOGRAM_FACTORY_HPP
#define HISTOGRAM_FACTORY_HPP

#include <map>
#include <vector>
#include <pcl_types.h>
#include <memory>
#include <Eigen/Core>

#include <SupervoxelSet.h>

namespace image_processing {

class HistogramFactory {

public:
    HistogramFactory(){
        _histograms.resize(4);
        _size = 0;
    }
    HistogramFactory(const pcl::Supervoxel<PointT>& sv, const std::vector<pcl::Supervoxel<PointT>>& neighbors){
        _histograms.resize(4);
        compute(sv,neighbors);
    }

    HistogramFactory(const HistogramFactory& HF) :
        _size(HF._size), _histogram_r(HF._histogram_r), _histogram_g(HF._histogram_g),
        _histogram_b(HF._histogram_b), _histogram_d(HF._histogram_d)
    {}

    /**
     * @brief compute the histograms related to sv
     * @param sv
     * @param (optional) neighbors of sv
     */
    void compute(const pcl::Supervoxel<PointT>& sv, const std::vector<pcl::Supervoxel<PointT>>& neighbors = std::vector<pcl::Supervoxel<PointT>>()){
        _size = neighbors.size() + 1;
        for(int i = 0; _histograms.size(); i++)
            _histograms[i].resize(_size);

        _histograms[0](0) = (double) sv.centroid_.r;
        _histograms[1](0) = (double) sv.centroid_.g;
        _histograms[2](0) = (double) sv.centroid_.b;
//        _histograms[3](0) = (double) sv.centroid_.z;

        for(int i = 1; i < _size; i++){
            _histograms[0](i) = fabs((double) sv.centroid_.r - _histograms[0](0));
            _histograms[1](i) = fabs((double) sv.centroid_.g - _histograms[1](0));
            _histograms[2](i) = fabs((double) sv.centroid_.b - _histograms[2](0));
            _histograms[3](i-1) = (double) sv.centroid_.z - _histograms[3](0);
        }
    }

    //GETTERS & SETTERS
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

};

}

#endif //HISTOGRAM_FACTORY_HPP
