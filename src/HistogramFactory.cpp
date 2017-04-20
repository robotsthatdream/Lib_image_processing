#include "image_processing/HistogramFactory.hpp"

using namespace image_processing;

void HistogramFactory::compute(const pcl::Supervoxel<image_processing::PointT>::Ptr &sv, std::string type){

    if(type == "color"){
        double r,g,b;
        float hsv[_dim];
        for(auto it = sv->voxels_->begin(); it != sv->voxels_->end(); ++it){
            r = it->r;
            g = it->g;
            b = it->b;
            tools::rgb2hsv(r,g,b,hsv[0],hsv[1],hsv[2]);
            int bin;
            for(int i = 0; i < _dim; i++){
                bin = (hsv[i] - _bounds(i,0))/(_bounds(i,1)/_bins);
                _histogram[i](bin)++;
            }
        }
    }
}


double HistogramFactory::chi_squared_distance(const Eigen::VectorXd& hist1, const Eigen::VectorXd &hist2){

    assert(hist1.rows() == hist2.rows());

    double sum = 0;
    for(int bin1 = 0; bin1 < hist1.rows(); ++bin1){
        for(int bin2 = 0; bin2 < hist2.rows(); ++bin2){
            sum += (hist1(bin1) - hist2(bin2))*(hist1(bin1) - hist2(bin2))/
                    (hist1(bin1)+hist2(bin2));
        }
    }


    return sum;
}

