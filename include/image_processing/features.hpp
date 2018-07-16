#ifndef _FEATURES_HPP
#define _FEATURES_HPP

#include <iostream>
#include <functional>
#include <image_processing/HistogramFactory.hpp>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <image_processing/pcl_types.h>
#include <eigen3/Eigen/Eigen>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/features/moment_invariants.h>
#include <pcl/features/boundary.h>
#include <tbb/tbb.h>

namespace image_processing{

typedef std::function<void(const SupervoxelArray&, const AdjacencyMap&, SupervoxelSet::features_t&)> function_t;

struct features_fct{
    static std::map<std::string,function_t> create_map(){
        std::map<std::string,function_t> map;

        map.emplace("colorNormalHist",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                Eigen::VectorXd sample(3);
                sample << sv.second->centroid_.r,
                        sv.second->centroid_.g,
                        sv.second->centroid_.b;
                features[sv.first]["colorRGB"] = sample;
            }
        });

        map.emplace("colorH",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
            for(const auto& sv : supervoxels){
                Eigen::VectorXd new_s(3);
                new_s << sv.second->normal_.normal[0],
                        sv.second->normal_.normal[1],
                        sv.second->normal_.normal[2];
                features[sv.first]["normal"] = new_s;
            }
        });

        map.emplace("normalX",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
            for(const auto& sv: supervoxels){
                std::vector<Eigen::VectorXd> data;
                for(auto it = sv.second->normals_->begin(); it != sv.second->normals_->end(); ++it){
                    Eigen::VectorXd vect(3);
                    vect(0) = it->normal[0];
                    vect(1) = it->normal[1];
                    vect[2] = it->normal[2];
                    data.push_back(vect);
                }

                Eigen::MatrixXd bounds(2,3);
                bounds << -1,-1,-1,
                        1,1,1;
                HistogramFactory hf(3,3,bounds);
                hf.compute_multi_dim(data);

                features[sv.first]["normalHistLarge"] = hf.get_histogram()[0];
            }
        });

        map.emplace("normalHistNeigh",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap& adj_map, SupervoxelSet::features_t& features){
            Eigen::VectorXd sum = Eigen::VectorXd::Zero(8);
            Eigen::MatrixXd bounds(2,3);
            bounds << -1,-1,-1,
                    1,1,1;
            HistogramFactory hf(2,3,bounds);
            int count;
            Eigen::VectorXd new_s(16);
            for(const auto& sv: supervoxels){
                std::vector<Eigen::VectorXd> data;
                auto it_pair = adj_map.equal_range(sv.first);
                count = 0;
                data.clear();
                for(auto it = it_pair.first; it != it_pair.second; it++){
                    for(auto it_norm = supervoxels.at(it->second)->normals_->begin();
                        it_norm != supervoxels.at(it->second)->normals_->end(); ++it_norm){
                        Eigen::VectorXd vect(3);
                        vect(0) = it_norm->normal[0];
                        vect(1) = it_norm->normal[1];
                        vect[2] = it_norm->normal[2];
                        data.push_back(vect);
                    }
                    hf.compute_multi_dim(data);
                    sum += hf.get_histogram()[0];
                    count++;
                }

                if(count > 0)
                    sum = sum/(float)count;



                data.clear();
                for(auto it = sv.second->normals_->begin(); it != sv.second->normals_->end(); ++it){
                    Eigen::VectorXd vect(3);
                    vect(0) = it->normal[0];
                    vect(1) = it->normal[1];
                    vect[2] = it->normal[2];
                    data.push_back(vect);
                }
                hf.compute_multi_dim(data);

                for(int i = 0; i < 8; i++)
                    new_s(i) = hf.get_histogram()[0](i);
                for(int i = 8; i < 16; i++){
                    if(fabs(sum(i-8)) < 1e-4)
                        sum(i-8) = 0.;
                    new_s(i) = sum(i - 8);
                }

                features[sv.first]["normalHistNeigh"] = new_s;
            }
        });

        map.emplace("colorLabHistLarge",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                hf.compute_multi_dim(data);

                features[sv.first]["colorLabHistLarge"] = hf.get_histogram()[0];
            }
        });



        map.emplace("fpfh",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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

        map.emplace("localMeanFPFH",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){

            pcl::FPFHEstimationOMP<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
            pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_cloud(new pcl::PointCloud<pcl::FPFHSignature33>);


            for(const auto& sv : supervoxels){
                fpfh.setInputCloud(sv.second->voxels_);
                fpfh.setInputNormals(sv.second->normals_);

                fpfh.setSearchMethod(tree);
                fpfh.setRadiusSearch (0.05);
                fpfh.compute(*fpfh_cloud);

                features[sv.first]["localMeanFPFH"] = Eigen::VectorXd::Zero(33);
                for(int i = 0; i < fpfh_cloud->size(); i++){
                    for(int j = 0; j < 33; j++)
                        features[sv.first]["localMeanFPFH"](j) += fpfh_cloud->points[i].histogram[j]/100.;
                }
                features[sv.first]["localMeanFPFH"] = features[sv.first]["localMeanFPFH"]/(double)fpfh_cloud->size();
            }
        });

        map.emplace("neighMeanFPFH",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap& adj_map, SupervoxelSet::features_t& features){
            std::vector<uint32_t> lbls;

            for(const auto& sv : supervoxels)
                lbls.push_back(sv.first);

            tbb::parallel_for(tbb::blocked_range<size_t>(0,lbls.size()),
                              [&](const tbb::blocked_range<size_t>& r){

                pcl::FPFHEstimationOMP<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
                pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
                pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_cloud(new pcl::PointCloud<pcl::FPFHSignature33>);
                PointCloudN::Ptr inputNormal(new PointCloudN);
                PointCloudT::Ptr inputCloud(new PointCloudT);
                pcl::IndicesPtr indices(new std::vector<int>);

                for(int k = r.begin(); k < r.end(); k++){
;
                    inputNormal.reset(new PointCloudN);
                    inputCloud.reset(new PointCloudT);
                    auto it_pair = adj_map.equal_range(lbls[k]);
                    for(auto it = it_pair.first; it != it_pair.second; it++){
                        if(supervoxels.find(it->second) == supervoxels.end())
                            continue;
                        for(int i = 0; i < supervoxels.at(it->second)->normals_->size(); i++){
                            inputNormal->push_back(supervoxels.at(it->second)->normals_->at(i));
                            inputCloud->push_back(supervoxels.at(it->second)->voxels_->at(i));
                        }
                    }

                    for(int i = 0; i < supervoxels.at(lbls[k])->normals_->size(); i++){
                        inputNormal->push_back(supervoxels.at(lbls[k])->normals_->at(i));
                        inputCloud->push_back(supervoxels.at(lbls[k])->voxels_->at(i));
                    }

                    indices.reset(new std::vector<int>);
                    for(int i = 0; i < inputCloud->size(); i++)
                        indices->push_back(i);

                    fpfh.setInputCloud(inputCloud);
                    fpfh.setInputNormals(inputNormal);
                    fpfh.setIndices(indices);

                    fpfh.setSearchMethod(tree);
                    fpfh.setRadiusSearch (0.05);
                    fpfh.compute(*fpfh_cloud);

                    features[lbls[k]]["neighMeanFPFH"] = Eigen::VectorXd::Zero(33);
                    for(int i = 0; i < fpfh_cloud->size(); i++){
                        for(int j = 0; j < 33; j++)
                            features[lbls[k]]["neighMeanFPFH"](j) += fpfh_cloud->points[i].histogram[j]/100.;
                    }
                    features[lbls[k]]["neighMeanFPFH"] = features[lbls[k]]["neighMeanFPFH"]/(double)fpfh_cloud->size();
                }
            });
       });

        map.emplace("meanFPFHLabHist",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap& adj_map, SupervoxelSet::features_t& features){

            std::vector<uint32_t> lbls;

            for(const auto& sv : supervoxels){
                lbls.push_back(sv.first);
                features.emplace(sv.first,std::map<std::string,Eigen::VectorXd>());
            }

            tbb::parallel_for(tbb::blocked_range<size_t>(0,lbls.size()),
                              [&](const tbb::blocked_range<size_t>& r){

                pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
                pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
                pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_cloud(new pcl::PointCloud<pcl::FPFHSignature33>);
                PointCloudN::Ptr inputNormal(new PointCloudN);
                PointCloudT::Ptr inputCloud(new PointCloudT);
                pcl::IndicesPtr indices(new std::vector<int>);
                Eigen::VectorXd new_s(48);
                for(int k = r.begin(); k < r.end(); k++){
//                for(int k = 0; k < lbls.size(); k++){
                    //* Lab
                    std::vector<Eigen::VectorXd> data;
                    for(auto it = supervoxels.at(lbls[k])->voxels_->begin(); it != supervoxels.at(lbls[k])->voxels_->end(); ++it){
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

                    int t = 0 , l = 0;
                    for(int i = 0; i < 15; i++){
                        new_s(i) = hf.get_histogram()[t](l);
                        l = (l+1)%5;
                        if(l == 0)
                            t++;
                    }
                    //*/

                    //* FPFH
                    inputNormal.reset(new PointCloudN);
                    inputCloud.reset(new PointCloudT);
                    auto it_pair = adj_map.equal_range(lbls[k]);
                    for(auto it = it_pair.first; it != it_pair.second; it++){
                        if(supervoxels.find(it->second) == supervoxels.end())
                            continue;
                        for(int i = 0; i < supervoxels.at(it->second)->normals_->size(); i++){
                            inputNormal->push_back(supervoxels.at(it->second)->normals_->at(i));
                            inputCloud->push_back(supervoxels.at(it->second)->voxels_->at(i));
                        }
                    }

                    for(int i = 0; i < supervoxels.at(lbls[k])->normals_->size(); i++){
                        inputNormal->push_back(supervoxels.at(lbls[k])->normals_->at(i));
                        inputCloud->push_back(supervoxels.at(lbls[k])->voxels_->at(i));
                    }

                    indices.reset(new std::vector<int>);
                    for(int i = 0; i < inputCloud->size(); i++)
                        indices->push_back(i);

                    fpfh.setInputCloud(inputCloud);
                    fpfh.setInputNormals(inputNormal);
                    fpfh.setIndices(indices);

                    fpfh.setSearchMethod(tree);
                    fpfh.setRadiusSearch (0.05);
                    fpfh.compute(*fpfh_cloud);

                    Eigen::VectorXd tmp = Eigen::VectorXd::Zero(33);
                    for(int i = 0; i < fpfh_cloud->size(); i++){
                        for(int j = 0; j < 33; j++)
                            tmp(j) += fpfh_cloud->points[i].histogram[j]/100.;
                    }
                    tmp = tmp/(double)fpfh_cloud->size();
                    for(int i = 0; i < 33; i++)
                        new_s(i+15) = tmp(i);
                    //*/

                    for(int i = 0; i < 48; i++){
                        if(new_s(i) > 1)
                            new_s(i) = 1;
                        else if (new_s(i) < 10e-4)
                            new_s(i) = 0;
                    }
                    features[lbls[k]]["meanFPFHLabHist"] = new_s;
                }
            });
        });

        map.emplace("circleFPFHLabHist",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap& adj_map, SupervoxelSet::features_t& features){

            std::vector<uint32_t> lbls;

            for(const auto& sv : supervoxels){
                lbls.push_back(sv.first);
                features.emplace(sv.first,std::map<std::string,Eigen::VectorXd>());
            }

            tbb::parallel_for(tbb::blocked_range<size_t>(0,lbls.size()),
                              [&](const tbb::blocked_range<size_t>& r){

                pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
                pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
                pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_cloud(new pcl::PointCloud<pcl::FPFHSignature33>);
                PointCloudN::Ptr inputNormal(new PointCloudN);
                PointCloudT::Ptr inputCloud(new PointCloudT);
                pcl::IndicesPtr indices(new std::vector<int>);
                Eigen::VectorXd new_s(48);
                for(int k = r.begin(); k < r.end(); k++){
//                for(int k = 0; k < lbls.size(); k++){
                    //* Lab
                    std::vector<Eigen::VectorXd> data;
                    for(auto it = supervoxels.at(lbls[k])->voxels_->begin(); it != supervoxels.at(lbls[k])->voxels_->end(); ++it){
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

                    int t = 0 , l = 0;
                    for(int i = 0; i < 15; i++){
                        new_s(i) = hf.get_histogram()[t](l);
                        l = (l+1)%5;
                        if(l == 0)
                            t++;
                    }
                    //*/

                    //* FPFH
                    inputNormal.reset(new PointCloudN);
                    inputCloud.reset(new PointCloudT);
                    double x,y,center_x,center_y;
                    center_x = supervoxels.at(k)->centroid_.x;
                    center_y = supervoxels.at(k)->centroid_.y;
                    for(const auto &sv : supervoxels){
                        for(int i = 0; sv.second->voxels_->size(); i++){
                            x = sv.second->voxels_->at(i).x;
                            y = sv.second->voxels_->at(i).y;
                            if((x-center_x)*(x-center_x) + (y - center_y)*(y - center_y) <= 0.2*0.2){
                                inputNormal->push_back(sv.second->normals_->at(i));
                                inputCloud->push_back(sv.second->voxels_->at(i));
                            }
                        }
                    }

                    indices.reset(new std::vector<int>);
                    for(int i = 0; i < inputCloud->size(); i++)
                        indices->push_back(i);

                    fpfh.setInputCloud(inputCloud);
                    fpfh.setInputNormals(inputNormal);
                    fpfh.setIndices(indices);

                    fpfh.setSearchMethod(tree);
                    fpfh.setRadiusSearch (0.05);
                    fpfh.compute(*fpfh_cloud);

                    Eigen::VectorXd tmp = Eigen::VectorXd::Zero(33);
                    for(int i = 0; i < fpfh_cloud->size(); i++){
                        for(int j = 0; j < 33; j++)
                            tmp(j) += fpfh_cloud->points[i].histogram[j]/100.;
                    }
                    tmp = tmp/(double)fpfh_cloud->size();
                    for(int i = 0; i < 33; i++)
                        new_s(i+15) = tmp(i);
                    //*/

                    for(int i = 0; i < 48; i++){
                        if(new_s(i) > 1)
                            new_s(i) = 1;
                        else if (new_s(i) < 10e-4)
                            new_s(i) = 0;
                    }
                    features[lbls[k]]["circleFPFHLabHist"] = new_s;
                }
            });
        });

        map.emplace("colorHSVNormal",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
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

        map.emplace("principalCurvatures",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap& adj_map, SupervoxelSet::features_t& features){
            pcl::PrincipalCurvaturesEstimation<PointT,pcl::Normal,pcl::PrincipalCurvatures> pce;
            Eigen::VectorXd new_s(5);
            std::vector<int> indices;
            pcl::KdTreeFLANN<PointT>::Ptr tree(new pcl::KdTreeFLANN<PointT>);
            std::vector<int> ind_tree(1);
            std::vector<float> dist_tree(1);
            float cx, cy, cz, c_min, c_max;
            PointCloudN::Ptr inputNormal(new PointCloudN);
            PointCloudT::Ptr inputCloud(new PointCloudT);

            for(const auto& sv : supervoxels){
                inputNormal->clear();
                inputCloud->clear();
                auto it_pair = adj_map.equal_range(sv.first);
                for(auto it = it_pair.first; it != it_pair.second; it++){
                    if(supervoxels.find(it->second) == supervoxels.end())
                        continue;
                    for(int i = 0; i < supervoxels.at(it->second)->normals_->size(); i++){
                        inputNormal->push_back(supervoxels.at(it->second)->normals_->at(i));
                        inputCloud->push_back(supervoxels.at(it->second)->voxels_->at(i));
                    }
                }

                for(int i = 0; i < sv.second->normals_->size(); i++){
                    inputNormal->push_back(sv.second->normals_->at(i));
                    inputCloud->push_back(sv.second->voxels_->at(i));
                }



                indices.clear();
                for(int i = 0; i < inputCloud->size(); i++)
                    indices.push_back(i);

                tree->setInputCloud(inputCloud);
                tree->nearestKSearch(sv.second->centroid_,1,ind_tree,dist_tree);
                pce.computePointPrincipalCurvatures(*inputNormal,ind_tree[0],indices,
                        cx,cy,cz,c_max,c_min);

                new_s << cx,cy,cz,c_max,c_min;

                for(int i = 0; i < new_s.rows(); i++)
                {
                    if(fabs(new_s(i)) < 1e-4)
                        new_s(i) = 0.;
                    else if(new_s(i) > 1.)
                        new_s(i) = 1.;
                    else if(new_s(i) < -1.)
                        new_s(i) = -1.;
                }

                features[sv.first]["principalCurvatures"] = new_s;
            }
        });

        map.emplace("prinCurvNeigh",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap& adj_map, SupervoxelSet::features_t& features){
            pcl::PrincipalCurvaturesEstimation<PointT,pcl::Normal,pcl::PrincipalCurvatures> pce;
            Eigen::VectorXd new_s(10);
            std::vector<int> indices;
            pcl::KdTreeFLANN<PointT>::Ptr tree(new pcl::KdTreeFLANN<PointT>);
            std::vector<int> ind_tree(1);
            std::vector<float> dist_tree(1);
            float cx, cy, cz, c_min, c_max;
            float cx2, cy2, cz2, c_min2, c_max2;
            float cx_n, cy_n, cz_n, c_min_n, c_max_n;
            int count;
            for(const auto& sv : supervoxels){
                count = 0;
                cx_n = 0; cy_n = 0; cz_n = 0; c_min_n = 0; c_max_n = 0;
                auto it_pair = adj_map.equal_range(sv.first);
                for(auto it = it_pair.first; it != it_pair.second; it++){
                    if(supervoxels.find(it->second) == supervoxels.end())
                        continue;
                    indices.clear();
                    for(int i = 0; i < sv.second->normals_->size(); i++)
                        indices.push_back(i);

                    tree->setInputCloud(sv.second->voxels_);
                    tree->nearestKSearch(sv.second->centroid_,1,ind_tree,dist_tree);
                    pce.computePointPrincipalCurvatures(*(sv.second->normals_),ind_tree[0],indices,
                            cx,cy,cz,c_max,c_min);

                    indices.clear();
                    for(int i = 0; i < supervoxels.at(it->second)->normals_->size(); i++)
                        indices.push_back(i);

                    tree->setInputCloud(supervoxels.at(it->second)->voxels_);
                    tree->nearestKSearch(supervoxels.at(it->second)->centroid_,1,ind_tree,dist_tree);
                    pce.computePointPrincipalCurvatures(*(supervoxels.at(it->second)->normals_),ind_tree[0],indices,
                            cx2,cy2,cz2,c_max2,c_min2);
                    cx_n+= fabs(cx-cx2);
                    cy_n+= fabs(cy-cy2);
                    cz_n+= fabs(cz-cz2);
                    c_min_n+= fabs(c_min-c_min2);
                    c_max_n+= fabs(c_max-c_max2);
                    count++;
                }
                if(count > 0){
                    cx_n = cx_n/(float)count;
                    cy_n = cy_n/(float)count;
                    cz_n = cz_n/(float)count;
                    c_min_n = c_min_n/(float)count;
                    c_max_n = c_max_n/(float)count;
                }



                new_s << cx, cy, cz, c_max, c_min, cx_n, cy_n, cz_n, c_max_n, c_min_n;


                for(int i = 0; i < new_s.rows(); i++)
                {
                    if(fabs(new_s(i)) < 1e-4)
                        new_s(i) = 0.;
                    else if(new_s(i) > 1.)
                        new_s(i) = 1.;
                    else if(new_s(i) < -1.)
                        new_s(i) = -1.;
                }

                features[sv.first]["prinCurvNeigh"] = new_s;
            }
        });

        map.emplace("centroidsPrinCurv",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
            pcl::PrincipalCurvaturesEstimation<PointT,pcl::Normal,pcl::PrincipalCurvatures> pce;
            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);


            Eigen::VectorXd new_s(5);

            PointCloudN::Ptr normals(new PointCloudN);
            PointCloudT::Ptr centroids(new PointCloudT);
            std::vector<uint32_t> lbl;

            pcl::PointCloud<pcl::PrincipalCurvatures> output_cloud;


            for(const auto& sv : supervoxels){
                normals->push_back(sv.second->normal_);
                centroids->push_back(sv.second->centroid_);
                lbl.push_back(sv.first);
            }

            pce.setSearchMethod(tree);
            pce.setRadiusSearch(0.1);
            pce.setInputNormals(normals);
            pce.setInputCloud(centroids);

            pce.compute(output_cloud);

            for(int i = 0; i < output_cloud.size(); i++){
                new_s << output_cloud[i].principal_curvature[0],
                        output_cloud[i].principal_curvature[1],
                        output_cloud[i].principal_curvature[2],
                        output_cloud[i].pc1, output_cloud[i].pc2;

                for(int i = 0; i < new_s.rows(); i++)
                {
                    if(new_s(i) != new_s(i))
                        new_s(i) = 0.;
                    else if(fabs(new_s(i)) < 1e-4)
                        new_s(i) = 0.;
                    else if(new_s(i) > 1.)
                        new_s(i) = 1.;
                    else if(new_s(i) < -1.)
                        new_s(i) = -1.;

                }

                features[lbl[i]]["centroidsPrinCurv"] = new_s;
            }

        });
        map.emplace("momentInvariant",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
            Eigen::VectorXd new_s(3);
            pcl::MomentInvariantsEstimation<PointT,pcl::MomentInvariants> mie;
            mie.setRadiusSearch(0.005);

            float j1,j2,j3;
            for(const auto& sv : supervoxels){
                mie.computePointMomentInvariants(*(sv.second->voxels_),j1,j2,j3);
                new_s << j1, j2, j3;
                features[sv.first]["momentInvariant"] = new_s;
            }
        });
        map.emplace("centroidsMomInv",
                    [](const SupervoxelArray& supervoxels, const AdjacencyMap&, SupervoxelSet::features_t& features){
            pcl::MomentInvariantsEstimation<PointT,pcl::MomentInvariants> mie;

            mie.setRadiusSearch(0.005);

            Eigen::VectorXd new_s(3);

            PointCloudT::Ptr centroids(new PointCloudT);
            std::vector<uint32_t> lbl;

            pcl::PointCloud<pcl::MomentInvariants> output_cloud;


            for(const auto& sv : supervoxels){
                centroids->push_back(sv.second->centroid_);
                lbl.push_back(sv.first);
            }

            mie.setInputCloud(centroids);

            mie.compute(output_cloud);

            for(int i = 0; i < output_cloud.size(); i++){
                new_s << output_cloud[i].j1,
                        output_cloud[i].j2,
                        output_cloud[i].j3;
                features[lbl[i]]["centroidsMomInv"] = new_s;
            }
        });

        map.emplace("localConvexityCP",
                   [](const SupervoxelArray& supervoxels, const AdjacencyMap& adj_map, SupervoxelSet::features_t& features){

            Eigen::Vector3d connectV, surfaceV,
                    centr_pos1, centr_pos2, norm1, norm2;
            Eigen::VectorXd new_s(8);
            double a1,a2;
            int count;
            for(const auto& sv : supervoxels){
                a1 = 0;
                a2 = 0;
                connectV = Eigen::Vector3d::Zero(3);
                surfaceV = Eigen::Vector3d::Zero(3);
                count = 0;
                centr_pos1 << sv.second->centroid_.x,
                        sv.second->centroid_.y,
                        sv.second->centroid_.z;
                norm1 << sv.second->normal_.normal[0],
                        sv.second->normal_.normal[1],
                        sv.second->normal_.normal[2];


                auto it_pair = adj_map.equal_range(sv.first);
                for(auto it = it_pair.first; it != it_pair.second; it++){
                    if(supervoxels.find(it->second) == supervoxels.end())
                        continue;
                    centr_pos2 << supervoxels.at(it->second)->centroid_.x,
                            supervoxels.at(it->second)->centroid_.y,
                            supervoxels.at(it->second)->centroid_.z;
                    norm2 << supervoxels.at(it->second)->normal_.normal[0],
                            supervoxels.at(it->second)->normal_.normal[1],
                            supervoxels.at(it->second)->normal_.normal[2];

                    connectV += centr_pos1 - centr_pos2;
                    surfaceV += norm1.cross(norm2);
                    a1 += norm1.dot(connectV);
                    a2 += norm1.dot(connectV);
                    count++;
                }
                a1 = a1/(double)count;
                a2 = a2/(double)count;
                connectV = connectV/(double)count;
                surfaceV = surfaceV/(double)count;

                new_s << connectV(0), connectV(1), connectV(2),
                        surfaceV(0), surfaceV(1), surfaceV(2),
                        a1, a2;

                for(int i = 0; i < new_s.rows(); i++)
                {
                    if(new_s(i) != new_s(i))
                        new_s(i) = 0.;
                    else if(fabs(new_s(i)) < 1e-4)
                        new_s(i) = 0.;
                    else if(new_s(i) > 1.)
                        new_s(i) = 1.;
                    else if(new_s(i) < -1.)
                        new_s(i) = -1.;
                }

                features[sv.first]["localConvexityCP"] = new_s;
            }

        });

        map.emplace("boundary",
                   [](const SupervoxelArray& supervoxels, const AdjacencyMap& adj_map, SupervoxelSet::features_t& features){
            pcl::BoundaryEstimation<PointT,pcl::Normal,pcl::Boundary> be;
            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
            PointCloudN::Ptr inputNormal(new PointCloudN);
            PointCloudT::Ptr inputCloud(new PointCloudT);
            PointCloudT boundaryCloud;
            pcl::PointCloud<pcl::Boundary> boundaries;
            for(const auto& sv: supervoxels){
                boundaries.clear();
                inputNormal->clear();
                inputCloud->clear();
                boundaryCloud.clear();

                auto it_pair = adj_map.equal_range(sv.first);
                for(auto it = it_pair.first; it != it_pair.second; it++){
                    if(supervoxels.find(it->second) == supervoxels.end())
                        continue;
                    for(int i = 0; i < supervoxels.at(it->second)->normals_->size(); i++){
                        inputNormal->push_back(supervoxels.at(it->second)->normals_->at(i));
                        inputCloud->push_back(supervoxels.at(it->second)->voxels_->at(i));
                    }
                }

                for(int i = 0; i < sv.second->normals_->size(); i++){
                    inputNormal->push_back(sv.second->normals_->at(i));
                    inputCloud->push_back(sv.second->voxels_->at(i));
                }

                be.setInputCloud(inputCloud);
                be.setInputNormals(inputNormal);
                be.setSearchMethod(tree);
                be.setRadiusSearch(0.02);
                be.compute(boundaries);

                for(int i = 0; i < boundaries.size(); i++){
                    if(boundaries[i].boundary_point != boundaries[i].boundary_point)
                        continue;
                    if(boundaries[i].boundary_point){
                        boundaryCloud.push_back(inputCloud->at(i));
                    }
                }
            }
        });
        return map;
    }



    static const std::map<std::string,function_t> fct_map;
};

}

#endif // _FEATURE_HPP
