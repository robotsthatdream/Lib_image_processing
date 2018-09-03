#include <iostream>

#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "../include/image_processing/SurfaceOfInterest.h"
#include <boost/archive/text_iarchive.hpp>
#include <iagmm/gmm.hpp>

namespace ip = image_processing;

pcl::PointCloud<pcl::PointXYZRGB>
getColoredWeightedCloud(ip::SurfaceOfInterest &soi, const std::string &modality,
                        int lbl, std::vector<std::set<uint32_t>> obj_indexes) {

    pcl::PointCloud<pcl::PointXYZRGB> result;
    pcl::PointXYZRGB pt;

    /* Draw all points in dark blueish tint, to see overall scene. */
    auto input_cloud = soi.getInputCloud();
    for (auto it_p = input_cloud->begin(); it_p != input_cloud->end(); it_p++) {
        // auto current_p = it_p->second;
        pt.x = it_p->x;
        pt.y = it_p->y;
        pt.z = it_p->z;

        pt.r = it_p->r / 8;
        pt.g = it_p->g / 4;
        pt.b = (it_p->r + it_p->g + it_p->b) / 6;
        result.push_back(pt);
    }

    auto supervoxels = soi.getSupervoxels();
    auto weights_for_this_modality = soi.get_weights()[modality];

    /* Draw all supervoxels points in various colors. */

    boost::random::mt19937 _gen;
    boost::random::uniform_int_distribution<> dist(4, 7);
    // _gen.seed(0); No seed, we want it deterministic.

    for (auto it_sv = supervoxels.begin(); it_sv != supervoxels.end();
         it_sv++) {
        int current_sv_label = it_sv->first;
        pcl::Supervoxel<ip::PointT>::Ptr current_sv = it_sv->second;
        float c = weights_for_this_modality[it_sv->first][lbl];

        if (c < 0.5) {
            std::cout << " skipping sv of label " << current_sv_label
                      << " weight " << c << std::endl;
            continue;
        } else {
            std::cout << " KEEPING sv of label " << current_sv_label
                      << " weight " << c << std::endl;
        }

        // Colors between quarter and half the max.  Not too weak, not too
        // bright.
        int r = float(dist(_gen) << 3) * (c + 1.0);
        int g = float(dist(_gen) << 3) * (c + 1.0);
        int b = float(dist(_gen) << 3) * (c + 1.0);

        for (auto v : *(current_sv->voxels_)) {
            pt.x = v.x;
            pt.y = v.y;
            pt.z = v.z;
            pt.r = r;
            pt.g = g;
            pt.b = b;
            result.push_back(pt);
        }
    }

    /* Populate again with cloud fitted with shape. */
    for (auto it_sv = supervoxels.begin(); it_sv != supervoxels.end();
         it_sv++) {
        pcl::Supervoxel<ip::PointT>::Ptr current_sv = it_sv->second;
        float c = weights_for_this_modality[it_sv->first][lbl];

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*(current_sv->voxels_), *cloud_xyz);
        std::vector<int> inliers;

        pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr model_s(
            new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(cloud_xyz));

        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_s);
        ransac.setDistanceThreshold(.001);
        ransac.computeModel();
        ransac.getInliers(inliers);

        Eigen::VectorXf coeff;
        ransac.getModelCoefficients(coeff);

        std::cerr << "coeff: " << coeff << std::endl;

        Eigen::VectorXf coeff_refined;
        model_s->optimizeModelCoefficients(inliers, coeff, coeff_refined);
        // EXPECT_EQ (4, coeff_refined.size ());

        pcl::PointCloud<pcl::PointXYZ> proj_points;
        model_s->projectPoints(inliers, coeff_refined, proj_points, false);

        for (auto v : proj_points) {
            pt.x = v.x;
            pt.y = v.y;
            pt.z = v.z;
            //            pt.rgb = (c * 255) * 0x010101;
            pt.r = pt.b = c * 255.0;
            pt.g = 0;
            result.push_back(pt);
        }

        // copies all inliers of the model computed to another PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr final(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud<pcl::PointXYZ>(*cloud_xyz, inliers, * final);

        for (auto v : *(final)) {
            pt.x = v.x;
            pt.y = v.y;
            pt.z = v.z;
            //            pt.rgb = (c * 255) * 0x010101;
            pt.r = pt.b = 0;
            pt.g = c * 255.0;
            result.push_back(pt);
        }
    }

    return result;
}

int main(int argc, char **argv) {

    if (argc != 3) {
        std::cerr << "Usage : \n\t- pcd file\n\t- gmm archive" << std::endl;
        return 1;
    }

    std::string pcd_file = argv[1];
    std::string gmm_archive = argv[2];

    //* Load pcd file into a pointcloud
    ip::PointCloudT::Ptr input_cloud(new ip::PointCloudT);
    pcl::io::loadPCDFile(pcd_file, *input_cloud);
    //*/

    std::cout << "pcd file loaded" << std::endl;

    //* Load the CMMs classifier from the archive
    std::ifstream ifs(gmm_archive);
    if (!ifs) {
        std::cerr << "Unable to open archive : " << gmm_archive << std::endl;
        return 1;
    }
    iagmm::GMM gmm;
    boost::archive::text_iarchive iarch(ifs);
    iarch >> gmm;
    //*/

    std::cout << "classifier archive loaded" << std::endl;

    //* Generate relevance map on the pointcloud
    ip::SurfaceOfInterest soi(input_cloud);
    std::cout << "computing supervoxel" << std::endl;
    soi.computeSupervoxel();

    std::cout << soi.getSupervoxels().size() << " supervoxels extracted"
              << std::endl;

    std::cout << "computed supervoxel" << std::endl;
    std::cout << "computing meanFPFHLabHist" << std::endl;
    soi.compute_feature("meanFPFHLabHist");
    std::cout << "computed meanFPFHLabHist" << std::endl;
    std::cout << "computing meanFPFHLabHist weights" << std::endl;
    soi.compute_weights<iagmm::GMM>("meanFPFHLabHist", gmm);
    std::cout << "computed meanFPFHLabHist weights" << std::endl;
    //*/

    std::cout << "relevance_map extracted" << std::endl;

    //* Generate objects hypothesis
    std::vector<std::set<uint32_t>> obj_indexes;
    obj_indexes = soi.extract_regions("meanFPFHLabHist", 0.5, 1);
    //*/

    std::cout << obj_indexes.size() << " objects hypothesis extracted"
              << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB> relevance_map_cloud =
        getColoredWeightedCloud(soi, "meanFPFHLabHist", 1, obj_indexes);

    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>
        relevance_map_cloud_ptr(&relevance_map_cloud);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer("Coucou 3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZRGB>(relevance_map_cloud_ptr, "cloud");

    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
    return 0;
}
