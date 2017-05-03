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
            double bin;
            for(int i = 0; i < _dim; i++){
                bin = (hsv[i] - _bounds(0,i))/(_bounds(1,i)/_bins);
                if(bin >= _bins) bin -= 1;
                _histogram[i](std::trunc(bin))++;
            }
        }
        for(int i = 0; i < _dim; i++){
            for(int j = 0; j < _bins; j++){
                _histogram[i](j) = _histogram[i](j)/((double)sv->voxels_->size());
            }
        }
    }
    if(type == "normal"){
        double normal[_dim];
        for(auto it = sv->normals_->begin(); it != sv->normals_->end(); ++it){
            normal[0] = it->normal[0];
            normal[1] = it->normal[1];
            normal[2] = it->normal[2];

            double bin;
            for(int i = 0; i < _dim; i++){
                bin = (normal[i] - _bounds(0,i))/(_bounds(1,i)/_bins);
                if(bin >= _bins) bin -= 1;
                _histogram[i](std::trunc(bin))++;
            }
        }
        for(int i = 0; i < _dim; i++){
            for(int j = 0; j < _bins; j++){
                _histogram[i](j) = _histogram[i](j)/((double)sv->voxels_->size());
            }
        }
    }
}


void HistogramFactory::compute(const cv::Mat& image){
    int image_chan = image.channels();
    uchar rgb[_dim];
    double bin;
    for(int i = 0; i < image.rows; i++){
        uchar* image_rowPtr = reinterpret_cast<uchar*>(image.row(i).data);
        for(int j = 0; j < image.cols; j++){
            rgb[0] = image_rowPtr[j*image_chan + 2];
            rgb[1] = image_rowPtr[j*image_chan + 1];
            rgb[2] = image_rowPtr[j*image_chan + 0];

            for(int i = 0; i < _dim; i++){
                bin = (rgb[i] - _bounds(0,i))/(_bounds(1,i)/_bins);
                if(bin >= _bins) bin -= 1;
                _histogram[i](std::trunc(bin))++;
            }
        }
    }
    for(int i = 0; i < _dim; i++){
        for(int j = 0; j < _bins; j++){
            _histogram[i](j) = _histogram[i](j)/(image.rows*image.cols);
        }
    }

}


double HistogramFactory::chi_squared_distance(const Eigen::VectorXd& hist1, const Eigen::VectorXd &hist2){

    assert(hist1.rows() == hist2.rows());

    double sum = 0;
    for(int bin = 0; bin < hist1.rows(); ++bin){
//        for(int bin2 = 0; bin2 < hist2.rows(); ++bin2){
        if(hist1(bin)+hist2(bin) != 0)
            sum += (hist1(bin) - hist2(bin))*(hist1(bin) - hist2(bin))/
                    (hist1(bin)+hist2(bin));
//        }
    }


    return sum/2.;
}

