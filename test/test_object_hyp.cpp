#include <boost/assert.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <forward_list>
#include <iostream>
#include <pcl/common/transforms.h>
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
#include "test_rotation.hpp"
#include <boost/archive/text_iarchive.hpp>
#include <iagmm/gmm.hpp>
#include <math.h>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

namespace ip = image_processing;

using namespace fsg::matrixrotationangles;

namespace fsg {
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudT;
typedef fsg::PointCloudT::Ptr PointCloudTP;

/**
    center x,y,z,
    rotation angles yaw, pitch, roll,
    radii 1,2,3,
    exponent1, exponent 2

    Total 11 parameters
*/
struct SuperEllipsoidParameters {
    Eigen::VectorXf coeff;

    SuperEllipsoidParameters() : coeff(11){};

// https://en.wikipedia.org/wiki/X_Macro

#define ALL_SuperEllipsoidParameters_FIELDS                                    \
    FSGX(cen_x)                                                                \
    FSGX(cen_y)                                                                \
    FSGX(cen_z)                                                                \
    FSGX(rad_major)                                                            \
    FSGX(rad_middle)                                                           \
    FSGX(rad_minor)                                                            \
    FSGX(rot_yaw)                                                              \
    FSGX(rot_pitch)                                                            \
    FSGX(rot_roll)                                                             \
    FSGX(exp_1)                                                                \
    FSGX(exp_2)

    enum idx {
#define FSGX(name) name,
        ALL_SuperEllipsoidParameters_FIELDS
#undef FSGX
    };

#define FSGX(name)                                                             \
    void set_##name(float f) { coeff(idx::name) = f; };
    ALL_SuperEllipsoidParameters_FIELDS
#undef FSGX

#define FSGX(name)                                                             \
    float get_##name() const { return coeff(idx::name); };
        ALL_SuperEllipsoidParameters_FIELDS
#undef FSGX

        friend ostream &
        operator<<(ostream &os, const SuperEllipsoidParameters &sefc);

    pcl::PointCloud<pcl::PointXYZ>::Ptr toPointCloud();
};

ostream &operator<<(ostream &os, const SuperEllipsoidParameters &sefc) {
    os << "[SEFC "
       << "center=(" << sefc.get_cen_x() << "," << sefc.get_cen_y() << ","
       << sefc.get_cen_z() << "), "
       << "radii=(" << sefc.get_rad_major() << "," << sefc.get_rad_middle()
       << "," << sefc.get_rad_minor() << "), "
       << "yaw=" << sefc.get_rot_yaw() << ", "
       << "pitch=" << sefc.get_rot_pitch() << ", "
       << "roll=" << sefc.get_rot_roll() << ", "
       << "exp_1=" << sefc.get_exp_1() << ", "
       << "exp_2=" << sefc.get_exp_2() << ", "
       << "]";
    return os;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr SuperEllipsoidParameters::toPointCloud() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_step1(
        new pcl::PointCloud<pcl::PointXYZ>);

    std::cout << "Creating a point cloud for " << *this << std::endl;

    // We start by creating a superquadric at world center, not rotated.

    // Same conventions as in fsg::matrixrotationangles, which in
    // turn follow ROS.

    // - x forward
    // - y left
    // - z up
    //
    //          Z
    //
    //          |
    //        x |
    //         \|
    //   Y -----O

    float exp_1 = this->get_exp_1();
    float exp_2 = this->get_exp_1();

    // FIXME rename rad_*, major will not always be Z
    float dilatfactor_x = this->get_rad_minor();
    float dilatfactor_y = this->get_rad_middle();
    float dilatfactor_z = this->get_rad_major();

    pcl::PointXYZ pt;
    const float increment = M_PI_2 / 10;

    // Pitch is eta in Biegelbauer et al.
    for (float pitch = -M_PI_2; pitch < M_PI_2; pitch += increment) {

        pt.z = dilatfactor_z * pow(sin(pitch), exp_1);
        float cos_pitch_exp_1 = pow(cos(pitch), exp_1);

        // Yaw is omega in Biegelbauer et al.
        for (float yaw = -M_PI; yaw < M_PI; yaw += increment) {

            pt.x = dilatfactor_x * pow(cos(yaw), exp_2) * cos_pitch_exp_1;
            pt.y = dilatfactor_y * pow(sin(yaw), exp_2) * cos_pitch_exp_1;

            cloud_step1->push_back(pt);
        }
    }

    // Next rotate the point cloud.

    Eigen::Matrix3f rotmat;
    angles_to_matrix(get_rot_yaw(), get_rot_pitch(), get_rot_roll(), rotmat);

    std::cout << "rotmat=" << rotmat << std::endl;

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0, 0, 3, 3) << rotmat.transpose();
    // Eigen::Vector3f center;
    // center << this->get_cen_x(), this->get_cen_y(), this->get_cen_z();
    transform.block(0, 3, 3, 1) << this->get_cen_x(), this->get_cen_y(), this->get_cen_z();

    std::cout << "transform=" << transform << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_final(
        new pcl::PointCloud<pcl::PointXYZ>);

    // You can either apply transform_1 or transform_2; they are the same
    pcl::transformPointCloud(*cloud_step1, *cloud_final, transform);

    return cloud_final;
}
}

struct OptimizationFunctor : pcl::Functor<float> {
    /** Functor constructor
     * \param[in] source cloud
     * \param[in] indices the indices of data points to evaluate
     */
    OptimizationFunctor(const pcl::PointCloud<pcl::PointXYZ> &cloud,
                        const std::vector<int> &indices)
        : pcl::Functor<float>(indices.size()), cloud_(cloud),
          indices_(indices) {}

    /** Cost function to be minimized
     * \param[in] x the variables array
     * \param[out] fvec the resultant functions evaluations
     * \return 0
     */
    int operator()(const Eigen::VectorXf &param, Eigen::VectorXf &fvec) const {
        // FG_TRACE_THIS_SCOPE();

        // Extract center;
        ip::PointT cen;
        cen.x = param(fsg::SuperEllipsoidParameters::idx::cen_x);
        cen.y = param(fsg::SuperEllipsoidParameters::idx::cen_y);
        cen.z = param(fsg::SuperEllipsoidParameters::idx::cen_z);

        // Compute rotation matrix
        Eigen::Matrix3f rotmat;
        angles_to_matrix(param(fsg::SuperEllipsoidParameters::idx::rot_yaw),
                         param(fsg::SuperEllipsoidParameters::idx::rot_pitch),
                         param(fsg::SuperEllipsoidParameters::idx::rot_roll),
                         rotmat);

        const float two_over_exp_1 =
            2.0 / param(fsg::SuperEllipsoidParameters::idx::exp_1);
        const float two_over_exp_2 =
            2.0 / param(fsg::SuperEllipsoidParameters::idx::exp_2);
        float exp_2_over_exp_1 = two_over_exp_1 / two_over_exp_2;

        for (signed int i = 0; i < values(); ++i) {

            // Take current point;
            pcl::PointXYZ p = cloud_.points[indices_[i]];

            // Compute vector from center.
            Eigen::Vector3f v_raw(p.x - cen.x, p.y - cen.y, p.z - cen.z);

            // Rotate vector
            Eigen::Vector3f v_aligned = rotmat * v_raw;

            // TODO check major/middle/minor vs X,Y,Z...

            Eigen::Vector3f v_scaled;
            v_scaled << v_aligned(0) /
                            param(
                                fsg::SuperEllipsoidParameters::idx::rad_major),
                v_aligned(1) /
                    param(fsg::SuperEllipsoidParameters::idx::rad_middle),
                v_aligned(2) /
                    param(fsg::SuperEllipsoidParameters::idx::rad_minor);

            float term = pow(v_scaled(0), two_over_exp_2) +
                         pow(v_scaled(1), two_over_exp_2);

            float outside_if_over_1 =
                pow(term, exp_2_over_exp_1) + pow(v_scaled(2), two_over_exp_1);

            float deviation = fabs(outside_if_over_1 - 1);

            fvec[i] = deviation;
        }
        return (0);
    }

    const pcl::PointCloud<pcl::PointXYZ> &cloud_;
    const std::vector<int> &indices_;
};

typedef struct cloud_reg {
    const char *key;
    const fsg::PointCloudTP cloud;
    const char *const name;
    bool active; // code smell: tied to a specific viewer
} cloud_reg_t;

class Context {
    const pcl::visualization::PCLVisualizer::Ptr m_viewer;
    std::forward_list<cloud_reg_t> m_clouds;

  public:
    Context(pcl::visualization::PCLVisualizer::Ptr &viewer)
        : m_viewer(viewer), m_clouds(){};
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

    m_clouds.push_front(reg);
    updateInViewer(reg);
    //    cloud_reg_t *newreg = &m_clouds.front();
}

void Context::handleKeyboardEvent(
    const pcl::visualization::KeyboardEvent &event) {
    if (event.keyUp()) {
        const std::string keySym = event.getKeySym();
        std::cout << "Key pressed '" << keySym << "'" << std::endl;

        if (keySym.compare("twosuperior") == 0) {
            m_viewer->setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0);
        }

        for (auto &cr : m_clouds) {
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

    fsg::PointCloudTP input_cloud_ptr(new fsg::PointCloudT());

    fsg::PointCloudTP supervoxel_cloud_ptr(new fsg::PointCloudT());

    fsg::PointCloudTP object_hyps_cloud_ptr(new fsg::PointCloudT());

    fsg::PointCloudTP superellipsoids_cloud_ptr(new fsg::PointCloudT());

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
            // int current_sv_label = it_sv->first;
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

            {
                pcl::PointXYZRGB pt;

                for (auto v : *cloud_xyz) {
                    pt.x = v.x;
                    pt.y = v.y;
                    pt.z = v.z;

                    pt.r = r;
                    pt.g = g;
                    pt.b = b;
                    object_hyps_cloud_ptr->push_back(pt);
                }
            }

            pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
            feature_extractor.setInputCloud(cloud_xyz);
            // Minimize eccentricity computation.
            feature_extractor.setAngleStep(360);
            feature_extractor.compute();

            Eigen::Vector3f mass_center;
            feature_extractor.getMassCenter(
                mass_center); // FIXME should check return value

            Eigen::Vector3f major_vector, middle_vector, minor_vector;

            feature_extractor.getEigenVectors(major_vector, middle_vector,
                                              minor_vector);

            pcl::PointXYZ min_point_OBB;
            pcl::PointXYZ max_point_OBB;
            pcl::PointXYZ position_OBB;
            Eigen::Matrix3f rotational_matrix_OBB;
            feature_extractor.getOBB(
                min_point_OBB, max_point_OBB, position_OBB,
                rotational_matrix_OBB); // FIXME should check return value

            // From
            // http://pointclouds.org/documentation/tutorials/moment_of_inertia.php

            Eigen::Vector3f position(position_OBB.x, position_OBB.y,
                                     position_OBB.z);
            Eigen::Quaternionf quat(rotational_matrix_OBB);

            std::string obbId(obj_index_i_s + ":obb");

            std::cout << "will add obb with id: " << obbId << std::endl;

            viewer->addCube(position, quat, max_point_OBB.x - min_point_OBB.x,
                            max_point_OBB.y - min_point_OBB.y,
                            max_point_OBB.z - min_point_OBB.z, obbId);

            viewer->setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
                obbId);

            pcl::PointXYZ center(mass_center(0), mass_center(1),
                                 mass_center(2));
            pcl::PointXYZ x_axis(major_vector(0) + mass_center(0),
                                 major_vector(1) + mass_center(1),
                                 major_vector(2) + mass_center(2));
            pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0),
                                 middle_vector(1) + mass_center(1),
                                 middle_vector(2) + mass_center(2));
            pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0),
                                 minor_vector(1) + mass_center(1),
                                 minor_vector(2) + mass_center(2));

            viewer->addLine(center, x_axis, 1.0f, 0.0f, 0.0f,
                            obj_index_i_s + ":major eigen vector");
            viewer->addLine(center, y_axis, 0.0f, 1.0f, 0.0f,
                            obj_index_i_s + ":middle eigen vector");
            viewer->addLine(center, z_axis, 0.0f, 0.0f, 1.0f,
                            obj_index_i_s + ":minor eigen vector");

            fsg::SuperEllipsoidParameters fittingContext;

            fittingContext.set_cen_x(mass_center(0));
            fittingContext.set_cen_y(mass_center(1));
            fittingContext.set_cen_z(mass_center(2));

            fittingContext.set_rad_major(major_vector.norm());

            fittingContext.set_rad_middle(middle_vector.norm());

            fittingContext.set_rad_minor(minor_vector.norm());

            float yaw, pitch, roll;
            matrix_to_angles(rotational_matrix_OBB, yaw, pitch, roll);

            std::cout << "yaw=" << yaw << ", pitch=" << pitch
                      << ", roll=" << roll << std::endl;

            fittingContext.set_rot_yaw(yaw);
            fittingContext.set_rot_pitch(pitch);
            fittingContext.set_rot_roll(roll);

            fittingContext.set_exp_1(1);
            fittingContext.set_exp_2(1);

            std::cout << "Initial estimation : " << fittingContext << std::endl;

            std::vector<int> indices(cloud_xyz->size());
            for (size_t i = 0; i < cloud_xyz->size(); ++i) {
                indices[i] = i;
            }

            OptimizationFunctor functor(*cloud_xyz, indices);
            Eigen::NumericalDiff<OptimizationFunctor> num_diff(functor);
            Eigen::LevenbergMarquardt<Eigen::NumericalDiff<OptimizationFunctor>,
                                      float>
                lm(num_diff);
            int minimizationResult = lm.minimize(fittingContext.coeff);

            std::cout << "Minimization result: " << (int)minimizationResult
                      << std::endl;

            std::cout << "After minimization : " << fittingContext << std::endl;

            pcl::PointCloud<pcl::PointXYZ>::Ptr proj_points =
                fittingContext.toPointCloud();

            {
                pcl::PointXYZRGB pt;

                r = 255.0 - r / 2.0;
                g = 255.0 - g / 2.0;
                b = 255.0 - b / 2.0;
                for (auto v : *proj_points) {
                    pt.x = v.x;
                    pt.y = v.y;
                    pt.z = v.z;

                    pt.r = r;
                    pt.g = g;
                    pt.b = b;
                    superellipsoids_cloud_ptr->push_back(pt);
                }
            }

            std::cout << "End new obj hyp, id=" << obj_index_i_s << "."
                      << std::endl;
        }
    }

    cloud_reg_t clouds[] = {
        {"1", input_cloud_ptr, "input", true},
        {"2", supervoxel_cloud_ptr, "supervoxel", true},
        {"3", object_hyps_cloud_ptr, "object_hyps", true},
        {"4", superellipsoids_cloud_ptr, "superellipsoids", true},
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
