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

            PointCloudT::Ptr centroids(new PointCloudT);
            PointCloudN::Ptr centroids_n(new PointCloudN);
            std::map<uint32_t, int> centroids_lbl;

            int i = 0;
            for(auto it = supervoxels.begin()
                ; it != supervoxels.end(); it++){
                pcl::Supervoxel<PointT>::Ptr supervoxel = it->second;
                centroids->push_back(supervoxel->centroid_);
                centroids_n->push_back(supervoxel->normal_);
                centroids_lbl.insert(std::pair<uint32_t,int>(it->first,i));
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

                sample.resize(63);
                int k = 0 , l = 0;
                for(int i = 0; i < 15; i++){
                    sample(i) = hf_color.get_histogram()[k](l);
                    l = (l+1)%5;
                    if(l == 0)
                        k++;
                }
                l = 0; k = 0;
                for(int i = 15; i < 30; i++){
                    sample(i) = hf_normal.get_histogram()[k](l);
                    l = (l+1)%5;
                    if(l == 0)
                        k++;
                }

                for(int i = 30; i < 63; ++i){
                    sample(i) = fpfh_cloud->points[centroids_lbl[sv.first]].histogram[i]/100.;
                }

                features[sv.first]["merged"] = sample;
            }
        });

        map.emplace("colorNormalHist",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                Eigen::VectorXd sample;

                Eigen::MatrixXd bounds_c(2,3);
                bounds_c << 0,0,0,
                        1,1,1;

                Eigen::MatrixXd bounds_n(2,3);
                bounds_n << -1,-1,-1,
                        1,1,1;
                HistogramFactory hf_color(10,3,bounds_c);
                HistogramFactory hf_normal(5,3,bounds_n);
                hf_color.compute(sv.second);
                hf_normal.compute(sv.second,"normal");

                sample.resize(45);
                int k = 0 , l = 0;
                for(int i = 0; i < 30; i++){
                    sample(i) = hf_color.get_histogram()[k](l);
                    l = (l+1)%10;
                    if(l == 0)
                        k++;
                }
                l = 0; k = 0;
                for(int i = 30; i < 45; i++){
                    sample(i) = hf_normal.get_histogram()[k](l);
                    l = (l+1)%5;
                    if(l == 0)
                        k++;
                }
                features[sv.first]["colorNormalHist"] = sample;
            }
        });

        //TO DO
//        map.emplace("colorHSVHistContrast",
//                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){

//            for(const auto& sv: supervoxels){
//                Eigen::MatrixXd bounds(2,3);
//                bounds << 0,0,0,
//                        1,1,1;
//                HistogramFactory hf(10,3,bounds);
//                hf.compute(sv.second);

//                Eigen::VectorXd sample(30);
//                int k = 0 , l = 0;
//                for(int i = 0; i < 30; i++){
//                    sample(i) = hf.get_histogram()[k](l);
//                    l = (l+1)%10;
//                    if(l == 0)
//                        k++;
//                }
//                features[sv.first]["colorHist"] = sample;
//            }

//        });

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

        map.emplace("colorLab",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                float Lab[3];
                tools::rgb2Lab(sv.second->centroid_.r,
                               sv.second->centroid_.g,
                               sv.second->centroid_.b,
                               Lab[0],Lab[1],Lab[2]);
                Eigen::VectorXd sample(3);
                sample << Lab[0], Lab[1], Lab[2];
                features[sv.first]["colorLab"] = sample;
            }
        });

        map.emplace("colorLabHist",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
           for(const auto& sv: supervoxels){
               std::vector<Eigen::VectorXd> data;
               for(auto it = sv.second->voxels_->begin(); it != sv.second->voxels_->end(); ++it){
                   float Lab[3];
                   tools::rgb2Lab(it->r,it->g,it->b,Lab[0],Lab[1],Lab[2]);
                   Eigen::VectorXd vect(3);
                   vect(0) = Lab[0];
                   vect(1) = Lab[1];
                   vect[2] = Lab[2];
                   data.push_back(vect);
               }
               Eigen::MatrixXd bounds(2,3);
               bounds << 0,-1,-1,
                       1,1,1;
               HistogramFactory hf(5,3,bounds);
               hf.compute(data);

               Eigen::VectorXd sample(15);
               int k = 0 , l = 0;
               for(int i = 0; i < 15; i++){
                   sample(i) = hf.get_histogram()[k](l);
                   l = (l+1)%5;
                   if(l == 0)
                       k++;
               }
               features[sv.first]["colorLabHist"] = sample;
           }
        });

        map.emplace("colorL",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv: supervoxels){
                std::vector<Eigen::VectorXd> data;
                for(auto it = sv.second->voxels_->begin(); it != sv.second->voxels_->end(); ++it){
                    float Lab[3];
                    tools::rgb2Lab(it->r,it->g,it->b,Lab[0],Lab[1],Lab[2]);
                    Eigen::VectorXd vect(3);
                    vect(0) = Lab[0];
                    vect(1) = Lab[1];
                    vect[2] = Lab[2];
                    data.push_back(vect);
                }
                Eigen::MatrixXd bounds(2,3);
                bounds << 0,-1,-1,
                        1,1,1;
                HistogramFactory hf(5,3,bounds);
                hf.compute(data);

                Eigen::VectorXd sample(5);
                sample = hf.get_histogram()[0];
                features[sv.first]["colorL"] = sample;
            }
        });

        map.emplace("colora",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv: supervoxels){
                std::vector<Eigen::VectorXd> data;
                for(auto it = sv.second->voxels_->begin(); it != sv.second->voxels_->end(); ++it){
                    float Lab[3];
                    tools::rgb2Lab(it->r,it->g,it->b,Lab[0],Lab[1],Lab[2]);
                    Eigen::VectorXd vect(3);
                    vect(0) = Lab[0];
                    vect(1) = Lab[1];
                    vect[2] = Lab[2];
                    data.push_back(vect);
                }
                Eigen::MatrixXd bounds(2,3);
                bounds << 0,-1,-1,
                        1,1,1;
                HistogramFactory hf(5,3,bounds);
                hf.compute(data);

                Eigen::VectorXd sample(5);
                sample = hf.get_histogram()[1];
                features[sv.first]["colora"] = sample;
            }
        });
        map.emplace("colorb",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv: supervoxels){
                std::vector<Eigen::VectorXd> data;
                for(auto it = sv.second->voxels_->begin(); it != sv.second->voxels_->end(); ++it){
                    float Lab[3];
                    tools::rgb2Lab(it->r,it->g,it->b,Lab[0],Lab[1],Lab[2]);
                    Eigen::VectorXd vect(3);
                    vect(0) = Lab[0];
                    vect(1) = Lab[1];
                    vect[2] = Lab[2];
                    data.push_back(vect);
                }
                Eigen::MatrixXd bounds(2,3);
                bounds << 0,-1,-1,
                        1,1,1;
                HistogramFactory hf(5,3,bounds);
                hf.compute(data);

                Eigen::VectorXd sample(5);
                sample = hf.get_histogram()[2];
                features[sv.first]["colorb"] = sample;
            }
        });

        map.emplace("colorLabNormalHist",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
           for(const auto& sv: supervoxels){
               std::vector<Eigen::VectorXd> data;
               for(auto it = sv.second->voxels_->begin(); it != sv.second->voxels_->end(); ++it){
                   float Lab[3];
                   tools::rgb2Lab(it->r,it->g,it->b,Lab[0],Lab[1],Lab[2]);
                   data.push_back(Eigen::VectorXd(3));
                   data.back()[0] = Lab[0];
                   data.back()[1] = Lab[1];
                   data.back()[2] = Lab[2];
               }
               Eigen::MatrixXd bounds_c(2,3);
               Eigen::MatrixXd bounds_n(2,3);
               bounds_c << 0,-1,-1,
                       1,1,1;
               bounds_n << -1,-1,-1,
                       1,1,1;
               HistogramFactory hf_color(5,3,bounds_c);
               hf_color.compute(data);
               HistogramFactory hf_normal(5,3,bounds_n);
               hf_normal.compute(sv.second,"normal");

               Eigen::VectorXd sample(30);
               int k = 0 , l = 0;
               for(int i = 0; i < 15; i++){
                   sample(i) = hf_color.get_histogram()[k](l);
                   l = (l+1)%5;
                   if(l == 0)
                       k++;
               }
               k = 0; l = 0;
               for(int i = 15; i < 30; i++){
                   sample(i) = hf_normal.get_histogram()[k](l);
                   l = (l+1)%5;
                   if(l == 0)
                       k++;
               }
               features[sv.first]["colorLabNormalHist"] = sample;
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

        map.emplace("colorHist",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv: supervoxels){
                Eigen::MatrixXd bounds(2,3);
                bounds << 0,0,0,
                        1,1,1;
                HistogramFactory hf(10,3,bounds);
                hf.compute(sv.second);

                Eigen::VectorXd sample(30);
                int k = 0 , l = 0;
                for(int i = 0; i < 30; i++){
                    sample(i) = hf.get_histogram()[k](l);
                    l = (l+1)%10;
                    if(l == 0)
                        k++;
                }
                features[sv.first]["colorHist"] = sample;
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

        map.emplace("normalHist",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv: supervoxels){
                Eigen::MatrixXd bounds(2,3);
                bounds << -1,-1,-1,
                        1,1,1;
                HistogramFactory hf(5,3,bounds);
                hf.compute(sv.second,"normal");

                Eigen::VectorXd sample(15);
                int k = 0 , l = 0;
                for(int i = 0; i < 15; i++){
                    sample(i) = hf.get_histogram()[k](l);
                    l = (l+1)%5;
                    if(l == 0)
                        k++;
                }
                features[sv.first]["normalHist"] = sample;
            }
        });

        map.emplace("normalHistLarge",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv: supervoxels){
                Eigen::MatrixXd bounds(2,3);
                bounds << -1,-1,-1,
                        1,1,1;
                HistogramFactory hf(5,3,bounds);
                hf.compute(sv.second,"normal");

                features[sv.first]["normalHistLarge"] = hf.get_histogram()[0];
            }
        };

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

        map.emplace("colorHSVNormal",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                float hsv[3];
                tools::rgb2hsv(sv.second->centroid_.r,
                               sv.second->centroid_.g,
                               sv.second->centroid_.b,
                               hsv[0],hsv[1],hsv[2]);

                Eigen::VectorXd new_s(6);
                new_s << sv.second->normal_.normal[0],
                        sv.second->normal_.normal[1],
                        sv.second->normal_.normal[2]
                        , hsv[0], hsv[1], hsv[2];
                features[sv.first]["colorHSVNormal"] = new_s;
            }
        });

        map.emplace("colorRGBNormal",
                    [](const SupervoxelArray& supervoxels, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                Eigen::VectorXd new_s(6);
                new_s << sv.second->normal_.normal[0],
                        sv.second->normal_.normal[1],
                        sv.second->normal_.normal[2],
                        sv.second->centroid_.r,
                        sv.second->centroid_.g,
                        sv.second->centroid_.b;
                features[sv.first]["colorRGBNormal"] = new_s;
            }
        });

        return map;
    }


    static const std::map<std::string,function_t> fct_map;
};

}

#endif // _FEATURE_HPP
