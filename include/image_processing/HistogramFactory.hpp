#ifndef HISTOGRAM_FACTORY_HPP
#define HISTOGRAM_FACTORY_HPP

#include <map>
#include <vector>
#include <image_processing/pcl_types.h>
#include <memory>
#include <Eigen/Core>
#include <image_processing/tools.hpp>

#include "SupervoxelSet.h"

namespace image_processing {

class HistogramFactory {

public:

    typedef std::vector<Eigen::VectorXd> _histogram_t;

    HistogramFactory(int bins, int dim, Eigen::MatrixXd bounds) :
        _bins(bins), _dim(dim), _bounds(bounds){
        _histogram = _histogram_t(_dim,Eigen::VectorXd::Zero(_bins));
    }
    HistogramFactory(const HistogramFactory& HF)
        : _bins(HF._bins), _dim(HF._dim), _bounds(HF._bounds),
          _histogram(HF._histogram){}



    /**
     * @brief compute the <type> histogram related to sv
     * @param sv
     * @param type {"color","normal"}
     */
    void compute(const pcl::Supervoxel<PointT>::Ptr& sv, std::string type = "color");

    /**
     * @brief chi_squared_distance
     * @param hist1
     * @param hist2
     * @return vector of distances
     */
    static double chi_squared_distance(const Eigen::VectorXd& hist1, const Eigen::VectorXd &hist2);


    //GETTERS & SETTERS

    /**
     * @brief return the histogram
     * @return histogram
     */
     _histogram_t get_histogram(){
        return _histogram;
    }


private:
    _histogram_t _histogram;

    int _bins;
    int _dim;
    Eigen::MatrixXd _bounds;

};

}

#endif //HISTOGRAM_FACTORY_HPP
