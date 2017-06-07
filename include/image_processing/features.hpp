#ifndef _FEATURES_HPP
#define _FEATURES_HPP

#include <iostream>
#include <functional>
#include <image_processing/HistogramFactory.hpp>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <image_processing/pcl_types.h>
#include <eigen3/Eigen/Eigen>


namespace image_processing{

typedef std::function<Eigen::VectorXd(const pcl::Supervoxel<PointT>::ConstPtr)> function_t;

struct features_fct{
    static std::map<std::string,function_t> create_map(){
        std::map<std::string,function_t> map;
        map.emplace("merged",
                    [](const pcl::Supervoxel<PointT>::ConstPtr sv ) -> Eigen::VectorXd{
            Eigen::VectorXd sample;

            Eigen::MatrixXd bounds_c(2,3);
            bounds_c << 0,0,0,
                    1,1,1;

            Eigen::MatrixXd bounds_n(2,3);
            bounds_n << -1,-1,-1,
                    1,1,1;
            HistogramFactory hf_color(5,3,bounds_c);
            HistogramFactory hf_normal(5,3,bounds_n);
            hf_color.compute(sv);
            hf_normal.compute(sv,"normal");

            sample.resize(30);
            int k = 0 , l = 0;
            for(int i = 0; i < 15; i++){
                sample(i) = hf_color.get_histogram()[k](l);
                k = (k+1)%3;
                l = (l+1)%5;
            }
            l = 0; k = 0;
            for(int i = 15; i < 30; i++){
                sample(i) = hf_normal.get_histogram()[k](l);
                k = (k+1)%3;
                l = (l+1)%5;
            }
            return sample;
        });

        map.emplace("colorHSV",
                    [](const pcl::Supervoxel<PointT>::ConstPtr sv ) -> Eigen::VectorXd{
            float hsv[3];
            tools::rgb2hsv(sv->centroid_.r,
                           sv->centroid_.g,
                           sv->centroid_.b,
                           hsv[0],hsv[1],hsv[2]);
            Eigen::VectorXd sample(3);
            sample << hsv[0], hsv[1], hsv[2];
            return sample;
        });

        map.emplace("colorRGB",
                    [](const pcl::Supervoxel<PointT>::ConstPtr sv ) -> Eigen::VectorXd{
            Eigen::VectorXd sample(3);
            sample << sv->centroid_.r,
                    sv->centroid_.g,
                    sv->centroid_.b;
            return sample;
        });

        map.emplace("colorH",
                    [](const pcl::Supervoxel<PointT>::ConstPtr sv ) -> Eigen::VectorXd{
            Eigen::MatrixXd bounds(2,3);
            bounds << 0,0,0,
                    1,1,1;
            HistogramFactory hf(5,3,bounds);
            hf.compute(sv);
            return hf.get_histogram()[0];
        });

        map.emplace("colorS",
                    [](const pcl::Supervoxel<PointT>::ConstPtr sv ) -> Eigen::VectorXd{
            Eigen::MatrixXd bounds(2,3);
            bounds << 0,0,0,
                    1,1,1;
            HistogramFactory hf(5,3,bounds);
            hf.compute(sv);
            return hf.get_histogram()[1];
        });

        map.emplace("colorV",
                    [](const pcl::Supervoxel<PointT>::ConstPtr sv ) -> Eigen::VectorXd{
            Eigen::MatrixXd bounds(2,3);
            bounds << 0,0,0,
                    1,1,1;
            HistogramFactory hf(5,3,bounds);
            hf.compute(sv);
            return hf.get_histogram()[2];
        });

        map.emplace("normal",
                    [](const pcl::Supervoxel<PointT>::ConstPtr sv ) -> Eigen::VectorXd{
            Eigen::VectorXd new_s(3);
            new_s << sv->normal_.normal[0],
                    sv->normal_.normal[1],
                    sv->normal_.normal[2];

            return new_s;
        });

        map.emplace("normalX",
                    [](const pcl::Supervoxel<PointT>::ConstPtr sv ) -> Eigen::VectorXd{

            Eigen::MatrixXd bounds(2,3);
            bounds << -1,-1,-1,
                    1,1,1;
            HistogramFactory hf(5,3,bounds);
            hf.compute(sv,"normal");
            return hf.get_histogram()[0];
        });

        map.emplace("normalY",
                    [](const pcl::Supervoxel<PointT>::ConstPtr sv ) -> Eigen::VectorXd{

            Eigen::MatrixXd bounds(2,3);
            bounds << -1,-1,-1,
                    1,1,1;
            HistogramFactory hf(5,3,bounds);
            hf.compute(sv,"normal");
            return hf.get_histogram()[1];
        });

        map.emplace("normalZ",
                    [](const pcl::Supervoxel<PointT>::ConstPtr sv ) -> Eigen::VectorXd{

            Eigen::MatrixXd bounds(2,3);
            bounds << -1,-1,-1,
                    1,1,1;
            HistogramFactory hf(5,3,bounds);
            hf.compute(sv,"normal");
            return hf.get_histogram()[2];
        });

        return map;
    }


    static const std::map<std::string,function_t> fct_map;
};

}

#endif // _FEATURE_HPP
