#include "image_processing/HistogramFactory.hpp"

using namespace image_processing;

void HistogramFactory::compute(const pcl::Supervoxel<image_processing::PointT>::ConstPtr &sv, std::string type){
    _histogram = _histogram_t(_dim,Eigen::VectorXd::Zero(_bins));

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
                if(hsv[i] != hsv[i] || (fabs(hsv[i]) > 10e3))
                    continue;
                if(fabs(hsv[i]) <= 10e-4)
                    hsv[i] = 0;

                bin = (hsv[i] - _bounds(0,i))/((_bounds(1,i) - _bounds(0,i))/_bins);
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
                if(normal[i] != normal[i] || (fabs(normal[i]) > 10e3))
                    continue;
                if(fabs(normal[i]) <= 10e-4)
                    normal[i] = 0;

                bin = (normal[i] - _bounds(0,i))/((_bounds(1,i) - _bounds(0,i))/_bins);
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
    _histogram = _histogram_t(_dim,Eigen::VectorXd::Zero(_bins));

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
                if(rgb[i] != rgb[i] || (fabs(rgb[i]) > 10e3))
                    continue;
                if(fabs(rgb[i]) <= 10e-4)
                    rgb[i] = 0;

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

void HistogramFactory::compute(const std::vector<Eigen::VectorXd>& data){
    _histogram = _histogram_t(_dim,Eigen::VectorXd::Zero(_bins));

    double bin, val;
    for(const auto& v: data){
        for(int i = 0; i < _dim; ++i){
            val = v[i];
            if(val != val  || (fabs(val) > 10e3))
                continue;
            if(fabs(val) <= 10e-4)
                val = 0;

            bin = (val - _bounds(0,i))/((_bounds(1,i) - _bounds(0,i))/_bins);
            if(bin >= _bins) bin -= 1;
            _histogram[i](std::trunc(bin))++;
        }
    }
    for(int i = 0; i < _dim; i++){
        for(int j = 0; j < _bins; j++){
            _histogram[i](j) = _histogram[i](j)/((double)data.size());
        }
    }
}

void HistogramFactory::compute_multi_dim(const std::vector<Eigen::VectorXd>& data){


    double bin[_dim],val;
    int d = 1, index = 0;
    for(int i = 0; i < _dim; i++) d = d*_bins;
    _histogram = _histogram_t(1,Eigen::VectorXd::Zero(d));
    for(auto& v: data){
        index = 0;
        for(int i = 0; i < _dim; i++){
            val = v[i];
            if(val != val  || (fabs(val) > 10e3))
                continue;
            if(fabs(val) <= 10e-4)
                val = 0;

            bin[i] = (val - _bounds(0,i))/((_bounds(1,i) - _bounds(0,i))/_bins);
            if(bin[i] >= _bins) bin[i] -= 1;
            int n = std::trunc(bin[i]);
//            for(int k = 0; k < n-1 ; k++){
//                int fact = 1;
//                for(int j = 0; j < i; j++)
//                    fact = fact * _bins;
//                index += fact;
//            }
            index += n*std::pow(_bins,i);
        }

        _histogram[0](index)++;
    }
    for(int j = 0; j < d; j++){
        _histogram[0](j) = _histogram[0](j)/((double)data.size());
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

