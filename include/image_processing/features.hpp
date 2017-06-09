#ifndef _FEATURES_HPP
#define _FEATURES_HPP

#include <iostream>
#include <functional>
#include <image_processing/HistogramFactory.hpp>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <image_processing/pcl_types.h>
#include <eigen3/Eigen/Eigen>
#include <pcl/features/fpfh.h>


namespace image_processing{

typedef std::function<void(const SupervoxelArray&, SupervoxelSet::features_t&)> function_t;

struct features_fct{
    static std::map<std::string,function_t> create_map(){
        std::map<std::string,function_t> map;
        map.emplace("merged",
                   [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                Eigen::VectorXd sample;

                Eigen::MatrixXd bounds_c(2,3);
                bounds_c << 0,0,0,
                        1,1,1;

                Eigen::MatrixXd bounds_n(2,3);
                bounds_n << -1,-1,-1,
                        1,1,1;
                HistogramFactory hf_color(5,3,bounds_c);
                HistogramFactory hf_normal(5,3,bounds_n);
                hf_color.compute(sv.second);
                hf_normal.compute(sv.second,"normal");

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
                features[sv.first]["merged"] = sample;
            }
        });

        map.emplace("colorHSV",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                float hsv[3];
                tools::rgb2hsv(sv.second->centroid_.r,
                               sv.second->centroid_.g,
                               sv.second->centroid_.b,
                               hsv[0],hsv[1],hsv[2]);
                Eigen::VectorXd sample(3);
                sample << hsv[0], hsv[1], hsv[2];
                features[sv.first]["colorHSV"] = sample;
            }
        });

        map.emplace("colorRGB",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                Eigen::VectorXd sample(3);
                sample << sv.second->centroid_.r,
                        sv.second->centroid_.g,
                        sv.second->centroid_.b;
                features[sv.first]["colorRGB"] = sample;
            }
        });

        map.emplace("colorH",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                Eigen::MatrixXd bounds(2,3);
                bounds << 0,0,0,
                        1,1,1;
                HistogramFactory hf(5,3,bounds);
                hf.compute(sv.second);
                features[sv.first]["colorH"] = hf.get_histogram()[0];
            }
        });

        map.emplace("colorS",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                Eigen::MatrixXd bounds(2,3);
                bounds << 0,0,0,
                        1,1,1;
                HistogramFactory hf(5,3,bounds);
                hf.compute(sv.second);
                features[sv.first]["colorS"] = hf.get_histogram()[1];
            }
        });

        map.emplace("colorV",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                Eigen::MatrixXd bounds(2,3);
                bounds << 0,0,0,
                        1,1,1;
                HistogramFactory hf(5,3,bounds);
                hf.compute(sv.second);
                features[sv.first]["colorV"] = hf.get_histogram()[2];
            }
        });

        map.emplace("normal",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                Eigen::VectorXd new_s(3);
                new_s << sv.second->normal_.normal[0],
                        sv.second->normal_.normal[1],
                        sv.second->normal_.normal[2];
                features[sv.first]["normal"] = new_s;
            }
        });

        map.emplace("normalX",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv: supervoxels){
                Eigen::MatrixXd bounds(2,3);
                bounds << -1,-1,-1,
                        1,1,1;
                HistogramFactory hf(5,3,bounds);
                hf.compute(sv.second,"normal");
                features[sv.first]["normalX"] = hf.get_histogram()[0];
            }
        });

        map.emplace("normalY",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv: supervoxels){
                Eigen::MatrixXd bounds(2,3);
                bounds << -1,-1,-1,
                        1,1,1;
                HistogramFactory hf(5,3,bounds);
                hf.compute(sv.second,"normal");
                features[sv.first]["normalY"] = hf.get_histogram()[1];
            }
        });

        map.emplace("normalZ",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv: supervoxels){
                Eigen::MatrixXd bounds(2,3);
                bounds << -1,-1,-1,
                        1,1,1;
                HistogramFactory hf(5,3,bounds);
                hf.compute(sv.second,"normal");
                features[sv.first]["normalZ"] = hf.get_histogram()[2];
            }
        });

        map.emplace("fpfh",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            PointCloudT::Ptr centroids(new PointCloudT);
            PointCloudN::Ptr centroids_n(new PointCloudN);
            std::map<int, uint32_t> centroids_lbl;

            int i = 0;
            for(auto it = supervoxels.begin()
                ; it != supervoxels.end(); it++){
                pcl::Supervoxel<PointT>::Ptr supervoxel = it->second;
                centroids->push_back(supervoxel->centroid_);
                centroids_n->push_back(supervoxel->normal_);
                centroids_lbl.insert(std::pair<int,uint32_t>(i,it->first));
                i++;
            }

            pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
            fpfh.setInputCloud(centroids);
            fpfh.setInputNormals(centroids_n);

            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
            fpfh.setSearchMethod(tree);
            fpfh.setRadiusSearch (0.05);


            pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_cloud(new pcl::PointCloud<pcl::FPFHSignature33>);
            fpfh.compute(*fpfh_cloud);
            for(auto feat : centroids_lbl){
                features[feat.second]["fpfh"] = Eigen::VectorXd(33);
                for(int i = 0; i < 33; ++i){
                   features[feat.second]["fpfh"](i) = fpfh_cloud->points[feat.first].histogram[i]/100.;
                }
            }
        });

        return map;
    }


    static const std::map<std::string,function_t> fct_map;
};

}

#endif // _FEATURE_HPP
