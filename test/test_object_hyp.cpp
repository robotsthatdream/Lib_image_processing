#include <boost/assert.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <forward_list>
#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/features/moment_of_inertia_estimation.h>
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

namespace fg {
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudT;
typedef fg::PointCloudT::Ptr PointCloudTP;
}

typedef struct cloud_reg {
    const char *key;
    const fg::PointCloudTP cloud;
    const char *const name;
    bool active; // code smell: tied to a specific viewer
} cloud_reg_t;

class Context {
    const pcl::visualization::PCLVisualizer::Ptr m_viewer;
    std::forward_list<cloud_reg_t> clouds;

  public:
    Context(pcl::visualization::PCLVisualizer::Ptr &viewer)
        : m_viewer(viewer), clouds(){};
    void addCloud(cloud_reg_t &reg);
    void handleKeyboardEvent(const pcl::visualization::KeyboardEvent &event);
    void updateInViewer(cloud_reg_t &cr);
};

void Context::updateInViewer(cloud_reg_t &cr) {
    m_viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_OPACITY, cr.active ? 1.0 : 0.0,
        cr.name);
}

void Context::addCloud(cloud_reg_t &reg) {
    std::cout << "Adding cloud with key " << reg.key << ", name " << reg.name
              << std::endl;

    m_viewer->addPointCloud<pcl::PointXYZRGB>(reg.cloud, reg.name);

    clouds.push_front(reg);
    updateInViewer(reg);
    //    cloud_reg_t *newreg = &clouds.front();
}

void Context::handleKeyboardEvent(
    const pcl::visualization::KeyboardEvent &event) {
    if (event.keyUp()) {
        const std::string keySym = event.getKeySym();
        std::cout << "Key pressed '" << keySym << "'" << std::endl;

        if (keySym.compare("twosuperior") == 0) {
            m_viewer->setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0);
        }

        for (auto &cr : clouds) {
            std::cout << "Checking key " << cr.key << ", name " << cr.name
                      << std::endl;
            if ((keySym.compare(cr.key) == 0)) {
                std::cout << "keysym " << keySym << " matches cloud name "
                          << cr.name << " => toggling (was " << cr.active << ")"
                          << std::endl;

                cr.active = !cr.active;
                updateInViewer(cr);
            }
        }
    }
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                           void *context_void) {
    boost::shared_ptr<Context> context =
        *static_cast<boost::shared_ptr<Context> *>(context_void);

    context->handleKeyboardEvent(event);
}

int main(int argc, char **argv) {

    if (argc != 4) {
        std::cerr << "Usage : \n\t- pcd file\n\t- gmm archive\n\t- label"
                  << std::endl;
        return 1;
    }

    std::string pcd_file = argv[1];
    std::string gmm_archive = argv[2];
    std::string label = argv[3];

    //* Load pcd file into a pointcloud
    ip::PointCloudT::Ptr input_cloud_soi(new ip::PointCloudT);
    pcl::io::loadPCDFile(pcd_file, *input_cloud_soi);
    //*/

    std::cout << "pcd file loaded:" << pcd_file << std::endl;

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

    std::cout << "classifier archive loaded:" << gmm_archive << std::endl;

    //* Generate relevance map on the pointcloud
    ip::SurfaceOfInterest soi(input_cloud_soi);
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
    std::vector<std::set<uint32_t>> obj_hypotheses;
    obj_hypotheses = soi.extract_regions("meanFPFHLabHist", 0.5, 1);
    //*/

    // obj_hypotheses

    std::cout << obj_hypotheses.size() << " objects hypothesis extracted"
              << std::endl;

    std::string windowTitle;

    {
        std::stringstream ss;
        ss << "Object fit viewer : " << label;
        windowTitle = ss.str();
    }

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer(label));
    viewer->setBackgroundColor(0, 0, 0);

    fg::PointCloudTP input_cloud_ptr(new fg::PointCloudT());

    fg::PointCloudTP supervoxel_cloud_ptr(new fg::PointCloudT());

    fg::PointCloudTP model_projected_all_cloud_ptr(new fg::PointCloudT());

    fg::PointCloudTP inliers_cloud_ptr(new fg::PointCloudT());

    {
        std::string modality = "meanFPFHLabHist";
        int lbl = 1;

        /* Draw all points in dark blueish tint, to see overall scene. */

        {
            pcl::PointXYZRGB pt;
            for (auto it_p = input_cloud_soi->begin();
                 it_p != input_cloud_soi->end(); it_p++) {
                // auto current_p = it_p->second;
                pt.x = it_p->x;
                pt.y = it_p->y;
                pt.z = it_p->z;

                pt.r = it_p->r;
                pt.g = it_p->g;
                pt.b = it_p->b; //(it_p->r + it_p->g + it_p->b) / 6;
                input_cloud_ptr->push_back(pt);
            }
        }

        ip::SupervoxelArray supervoxels = soi.getSupervoxels();
        ip::SurfaceOfInterest::relevance_map_t weights_for_this_modality =
            soi.get_weights()[modality];

        /* Draw all supervoxels points in various colors. */

        boost::random::mt19937 _gen;
        boost::random::uniform_int_distribution<> dist(1, 3);
        // _gen.seed(0); No seed, we want it deterministic.

        int kept = 0;
        for (auto it_sv = supervoxels.begin(); it_sv != supervoxels.end();
             it_sv++) {
            int current_sv_label = it_sv->first;
            pcl::Supervoxel<ip::PointT>::Ptr current_sv = it_sv->second;
            float c = weights_for_this_modality[it_sv->first][lbl];

            if (c < 0.5) {
                // std::cout << " skipping sv of label " << current_sv_label <<
                // " weight " << c << std::endl;
                continue;
            }
            // std::cout << " KEEPING sv of label " << current_sv_label << "
            // weight " << c << std::endl;
            ++kept;

            // This chooses random colors amont a palette of 3^3 = 27
            // different colors, then multiplied by biased relevance.
            // Hope this allows to easily distinguish supervoxels by
            // colors.
            int r = float(dist(_gen) * 56) * (c + 0.5);
            int g = float(dist(_gen) * 56) * (c + 0.5);
            int b = float(dist(_gen) * 56) * (c + 0.5);

            pcl::PointXYZRGB pt;
            for (auto v : *(current_sv->voxels_)) {
                pt.x = v.x;
                pt.y = v.y;
                pt.z = v.z;
                pt.r = r;
                pt.g = g;
                pt.b = b;
                supervoxel_cloud_ptr->push_back(pt);
            }
        }
        std::cout << "Thresholding kept " << kept << " supervoxels out of "
                  << supervoxels.size() << std::endl;

        /* Populate again with cloud fitted with shape. */

        /* We have to express what supervoxels belong together.

           We could copy points, or just set indices, which saves memory.
           Actually, PCL uses indices anyway.

           We don't have to filter again because extract_regions already does
           it.

        */

        // Rappel : typedef std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>
        // SupervoxelArray;

        /* each object */
        for (const auto &obj_hyp :
             obj_hypotheses | boost::adaptors::indexed(0)) {
            std::string obj_index_i_s = std::to_string(obj_hyp.index());
            std::set<uint32_t> *p_obj_hyp = &(obj_hyp.value());

            int r = float(dist(_gen) * 85);
            int g = float(dist(_gen) * 85);
            int b = float(dist(_gen) * 85);

            std::cout << std::endl
                      << "Begin new obj hyp, id=" << obj_index_i_s << ", "
                                                                      "color = "
                      << r << "," << g << "," << b << std::endl;

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(
                new pcl::PointCloud<pcl::PointXYZ>);

            if (p_obj_hyp->size() <= 1) {
                std::cerr << "Skipping hypothesis object id=" << obj_index_i_s
                          << " because too few supervoxels: "
                          << p_obj_hyp->size() << std::endl;
                continue;
            }

            {
                int kept = 0;
                pcl::PointXYZ pt;
                for (auto it_sv = supervoxels.begin();
                     it_sv != supervoxels.end(); it_sv++) {
                    int current_sv_label = it_sv->first;
                    pcl::Supervoxel<ip::PointT>::Ptr current_sv = it_sv->second;

                    if (p_obj_hyp->find(current_sv_label) == p_obj_hyp->end()) {
                        // std::cout << "Supervoxel " << current_sv_label << "
                        // not part of current object, skipping." << std::endl;
                        continue;
                    }
                    ++kept;

                    std::cout << "Supervoxel labelled " << current_sv_label
                              << " part of current object, including, "
                                 "will add "
                              << current_sv->voxels_->size() << " point(s)."
                              << std::endl;
                    for (auto v : *(current_sv->voxels_)) {
                        pt.x = v.x;
                        pt.y = v.y;
                        pt.z = v.z;
                        cloud_xyz->push_back(pt);
                    }
                }
                std::cout << "Gathered " << kept
                          << " supervoxels into a point cloud of size "
                          << cloud_xyz->size() << std::endl;
            }

            if (cloud_xyz->size() < 20) {
                std::cerr
                    << "Skipping hypothesis object id=" << obj_index_i_s
                    << " because supervoxels combined into too few points: "
                    << cloud_xyz->size() << std::endl;
                continue;
            }

            /* Okay, we've got one point cloud for this object. */

            pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
            feature_extractor.setInputCloud(cloud_xyz);
            // Minimize eccentricity computation.
            feature_extractor.setAngleStep(360);
            feature_extractor.compute();

            Eigen::Vector3f mass_center;
            feature_extractor.getMassCenter(
                mass_center); // FIXME should check return value

            std::vector<float> moment_of_inertia;
            pcl::PointXYZ min_point_OBB;
            pcl::PointXYZ max_point_OBB;
            pcl::PointXYZ position_OBB;
            Eigen::Matrix3f rotational_matrix_OBB;
            feature_extractor.getOBB(
                min_point_OBB, max_point_OBB, position_OBB,
                rotational_matrix_OBB); // FIXME should check return value


            
            pcl::PointCloud<pcl::PointXYZ> proj_points;
            // model_s->projectPoints(inliers, coeff_refined, proj_points,
            // false);

            pcl::PointXYZRGB pt;

            for (auto v : *cloud_xyz) {
                pt.x = v.x;
                pt.y = v.y;
                pt.z = v.z;

                pt.r = r;
                pt.g = g;
                pt.b = b;
                model_projected_all_cloud_ptr->push_back(pt);
            }

            std::cout << "End new obj hyp, id=" << obj_index_i_s << "."
                      << std::endl;
        }
    }

    cloud_reg_t clouds[] = {
        {"1", input_cloud_ptr, "input", true},
        {"2", supervoxel_cloud_ptr, "supervoxel", true},
        {"3", model_projected_all_cloud_ptr, "model_projected_all", true},
        {"4", inliers_cloud_ptr, "inliers", true},
        //{ "t", superellipsoid_cloud, "superellipsoid_cloud" },
    };

    boost::shared_ptr<Context> context_p(new Context(viewer));

    for (auto &cr : clouds) {
        context_p->addCloud(cr);
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, cr.name);
    }

    // viewer->addCoordinateSystem (1.0);
    viewer->setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0);

    viewer->registerKeyboardCallback(keyboardEventOccurred, (void *)&context_p);

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    viewer->close();

    return 0;
}
