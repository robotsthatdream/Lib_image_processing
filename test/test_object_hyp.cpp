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

#include <vtkOrientationMarkerWidget.h>
#include <vtkRenderWindow.h>

#include "../include/image_processing/SurfaceOfInterest.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#include "cmaes.h"
#pragma GCC diagnostic pop

#include "test_rotation.hpp"
#include <boost/archive/text_iarchive.hpp>
#include <iagmm/gmm.hpp>
#include <math.h>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include "fsg_trace.hpp"

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
// https://en.wikipedia.org/wiki/X_Macro

#define ALL_SuperEllipsoidParameters_FIELDS                                    \
    FSGX(cen_x)                                                                \
    FSGX(cen_y)                                                                \
    FSGX(cen_z)                                                                \
    FSGX(rad_a)                                                                \
    FSGX(rad_b)                                                                \
    FSGX(rad_c)                                                                \
    FSGX(rot_yaw)                                                              \
    FSGX(rot_pitch)                                                            \
    FSGX(rot_roll)                                                             \
    FSGX(exp_1)                                                                \
    FSGX(exp_2)

    static SuperEllipsoidParameters Zero() {
        SuperEllipsoidParameters zero;
#define FSGX(name) zero.set_##name(0.00);
        ALL_SuperEllipsoidParameters_FIELDS;
#undef FSGX
        return zero;
    }

    static SuperEllipsoidParameters Default() {
        SuperEllipsoidParameters dv = Zero();

        dv.set_rad_a(1.0);
        dv.set_rad_b(2.0);
        dv.set_rad_c(3.0);
        dv.set_exp_1(1.0);
        dv.set_exp_2(1.0);
        return dv;
    }

    enum idx {
#define FSGX(name) name,
        ALL_SuperEllipsoidParameters_FIELDS
#undef FSGX
    };

#define FSGX(name)                                                             \
    void set_##name(float f) { coeff(idx::name) = f; };
    ALL_SuperEllipsoidParameters_FIELDS;
#undef FSGX

#define FSGX(name)                                                             \
    float get_##name() const { return coeff(idx::name); };
    ALL_SuperEllipsoidParameters_FIELDS;
#undef FSGX

    static constexpr int fieldCount = 0
#define FSGX(name) +1
        ALL_SuperEllipsoidParameters_FIELDS
#undef FSGX
        ;

    // https://stackoverflow.com/questions/11490988/c-compile-time-error-expected-identifier-before-numeric-constant
    Eigen::VectorXd coeff = Eigen::VectorXd(fieldCount);

    double *coeffData();

    SuperEllipsoidParameters() : coeff(fieldCount){};

    friend ostream &operator<<(ostream &os,
                               const SuperEllipsoidParameters &sefc);

    pcl::PointCloud<pcl::PointXYZ>::Ptr toPointCloud(int steps);
};

constexpr int SuperEllipsoidParameters::fieldCount;

ostream &operator<<(ostream &os, const SuperEllipsoidParameters &sefc) {
    os << "[SEFC "
       << "center=(" << sefc.get_cen_x() << "," << sefc.get_cen_y() << ","
       << sefc.get_cen_z() << "), "
       << "radii=(" << sefc.get_rad_a() << "," << sefc.get_rad_b() << ","
       << sefc.get_rad_c() << "), "
       << "yaw=" << sefc.get_rot_yaw() << ", "
       << "pitch=" << sefc.get_rot_pitch() << ", "
       << "roll=" << sefc.get_rot_roll() << ", "
       << "exp_1=" << sefc.get_exp_1() << ", "
       << "exp_2=" << sefc.get_exp_2() << ", "
       << "]";
    return os;
}

float powf_sym(float x, float y) {
    if (std::signbit(x) != 0)
        return -powf(-x, y);
    else
        return powf(x, y);
}

/** Provide access to coeff data as C-style array.  Number of
    values is provided in SuperEllipsoidParameters::fieldcount.
 */
double *SuperEllipsoidParameters::coeffData() { return (this->coeff.data()); }

/** This method implements the forward transformation from a
    12-dimension model to a point cloud.

    Compare with the inverse transformation implemented in `struct
    OptimizationFunctor`.
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr
SuperEllipsoidParameters::toPointCloud(int steps) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_step1(
        new pcl::PointCloud<pcl::PointXYZ>);
    FSG_TRACE_THIS_FUNCTION();
    FSG_LOG_MSG("Creating a point cloud with " << steps << " steps for "
                                               << *this);

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
    float exp_2 = this->get_exp_2();

    float dilatfactor_x = this->get_rad_a();
    float dilatfactor_y = this->get_rad_b();
    float dilatfactor_z = this->get_rad_c();

    pcl::PointXYZ pt;
    const float increment = M_PI_2 / steps;

    // Pitch is eta in Biegelbauer et al.
    for (float pitch = -M_PI_2; pitch < M_PI_2; pitch += increment) {

        pt.z = dilatfactor_z * powf_sym(sin(pitch), exp_1);
        float cos_pitch_exp_1 = powf_sym(cos(pitch), exp_1);

        // Yaw is omega in Biegelbauer et al.
        for (float yaw = -M_PI; yaw < M_PI; yaw += increment) {

            pt.x = dilatfactor_x * powf_sym(cos(yaw), exp_2) * cos_pitch_exp_1;
            pt.y = dilatfactor_y * powf_sym(sin(yaw), exp_2) * cos_pitch_exp_1;

            if ((pt.x * pt.x + pt.y * pt.y + pt.z * pt.z) < 20.0) {
                cloud_step1->push_back(pt);
            } else {
                FSG_LOG_MSG("Ignoring too far point " << pt);
            }
        }
    }

    // Next rotate the point cloud.

    Eigen::Matrix3f rotmat;
    angles_to_matrix(get_rot_yaw(), get_rot_pitch(), get_rot_roll(), rotmat);

    FSG_LOG_VAR(rotmat);

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0, 0, 3, 3) << rotmat;
    // Eigen::Vector3f center;
    // center << this->get_cen_x(), this->get_cen_y(), this->get_cen_z();
    transform.block(0, 3, 3, 1) << this->get_cen_x(), this->get_cen_y(),
        this->get_cen_z();

    FSG_LOG_VAR(transform);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_final(
        new pcl::PointCloud<pcl::PointXYZ>);

    // You can either apply transform_1 or transform_2; they are the same
    pcl::transformPointCloud(*cloud_step1, *cloud_final, transform);

    return cloud_final;
}
} // namespace fsg

/** This method implements the inverse transformation, which computes
    from a point cloud its fitting to a candidate 12-dimension.

    Compare with the forward transformation implemented in
    `SuperEllipsoidParameters::toPointCloud`.
 */
struct OptimizationFunctor : pcl::Functor<float> {
    /** Functor constructor
     * \param[in] source cloud
     * \param[in] indices the indices of data points to evaluate
     */
    OptimizationFunctor(const pcl::PointCloud<pcl::PointXYZ> &cloud,
                        const std::vector<int> &indices)
        : pcl::Functor<float>(indices.size()), cloud_(cloud),
          indices_(indices) {
        FSG_LOG_MSG("Created functor with value count: " << values());
    }

#define powf_abs(x, y) powf(fabs(x), y)
// float powf_abs(const float x, const float y) const {
//     FSG_TRACE_THIS_FUNCTION();
//     FSG_LOG_VAR(x);
//     FSG_LOG_VAR(y);
//     const float absx = fabs(x);
//     FSG_LOG_VAR(absx);
//     const float result = powf(absx, y);
//     FSG_LOG_VAR(result);
//     return result;
// }

#define FUNCTOR_LOG_INSIDE 0

    /** Cost function to be minimized
     * \param[in] x the variables array
     * \param[out] fvec the resultant functions evaluations
     * \return 0
     */
    int operator()(const Eigen::VectorXd &param, Eigen::VectorXf &fvec) const {
#if FUNCTOR_LOG_INSIDE == 1
        fsg::SuperEllipsoidParameters *sep =
            (fsg::SuperEllipsoidParameters *)((void *)&param); // Yeww hack.
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM("f(): " << *sep);
#endif

        const float exp_1 = param(fsg::SuperEllipsoidParameters::idx::exp_1);
        // FSG_LOG_VAR(exp_1);
        const float exp_2 = param(fsg::SuperEllipsoidParameters::idx::exp_2);
        // FSG_LOG_VAR(exp_2);

        // if ((exp_1 > 2.0) || (exp_2 > 2.0)) {
        //     FSG_LOG_MSG("Not doing computation because too big exponent: 1:"
        //                 << exp_1 << " 2:" << exp_2);
        //     for (signed int i = 0; i < values(); ++i) {
        //         fvec[i] = -FLT_MAX;
        //     }
        //     return 0;
        // }

        // Extract center;
        pcl::PointXYZ cen;
        cen.x = param(fsg::SuperEllipsoidParameters::idx::cen_x);
        cen.y = param(fsg::SuperEllipsoidParameters::idx::cen_y);
        cen.z = param(fsg::SuperEllipsoidParameters::idx::cen_z);
        // FSG_LOG_VAR(cen);

        // Compute rotation matrix
        Eigen::Matrix3f rotmat;
        angles_to_matrix(param(fsg::SuperEllipsoidParameters::idx::rot_yaw),
                         param(fsg::SuperEllipsoidParameters::idx::rot_pitch),
                         param(fsg::SuperEllipsoidParameters::idx::rot_roll),
                         rotmat);
        // FSG_LOG_VAR(rotmat);
        rotmat.transposeInPlace();
        // FSG_LOG_VAR(rotmat);

        const float two_over_exp_1 = 2.0 / exp_1;
        const float two_over_exp_2 = 2.0 / exp_2;
        const float exp_2_over_exp_1 = exp_2 / exp_1;
        // FSG_LOG_VAR(two_over_exp_2);
        // FSG_LOG_VAR(two_over_exp_1);
        // FSG_LOG_VAR(exp_2_over_exp_1);

        float sum_of_squares = 0;

        for (signed int i = 0; i < values(); ++i) {
            // Take current point;
            const pcl::PointXYZ p = cloud_.points[indices_[i]];
            // FSG_LOG_VAR(p);

            // Compute vector from center.
            const Eigen::Vector3f v_raw(p.x - cen.x, p.y - cen.y, p.z - cen.z);
            // FSG_LOG_VAR(v_raw);

            // Rotate vector
            const Eigen::Vector3f v_aligned = rotmat * v_raw;
            // FSG_LOG_VAR(v_aligned);

            // TODO check major/middle/minor vs X,Y,Z...

            Eigen::Vector3f v_scaled;
            // FIXME radii here are not major middle minor, only x y z or 1 2 3.
            v_scaled << v_aligned(0) /
                            param(fsg::SuperEllipsoidParameters::idx::rad_a),
                v_aligned(1) / param(fsg::SuperEllipsoidParameters::idx::rad_b),
                v_aligned(2) / param(fsg::SuperEllipsoidParameters::idx::rad_c);
            // FSG_LOG_VAR(v_scaled);

            const float term = powf_abs(v_scaled(0), two_over_exp_2) +
                               powf_abs(v_scaled(1), two_over_exp_2);
            // FSG_LOG_VAR(term);

            const float outside_if_over_1 =
                powf_abs(term, exp_2_over_exp_1) +
                powf_abs(v_scaled(2), two_over_exp_1);
            FSG_LOG_VAR(outside_if_over_1);

            const float deviation = log(outside_if_over_1);
            // FSG_LOG_VAR(deviation);

            fvec[i] = deviation;
            sum_of_squares += deviation * deviation;
        }
// FSG_LOG_VAR(fvec);
#if FUNCTOR_LOG_INSIDE == 1
        FSG_LOG_VAR(sum_of_squares);
#endif
        return (sum_of_squares);
    }

    const pcl::PointCloud<pcl::PointXYZ> &cloud_;
    const std::vector<int> &indices_;
};

struct cloud_reg {
    const char *key;
    const fsg::PointCloudTP cloud;
    const char *const name;
    bool active; // code smell: tied to a specific viewer

    friend ostream &operator<<(ostream &os, const cloud_reg &sefc);
};

ostream &operator<<(ostream &os, const cloud_reg &cr) {
    os << "[cloud_reg " << FSG_OSTREAM_FIELD(cr, key)
       << FSG_OSTREAM_FIELD(cr, name) << FSG_OSTREAM_FIELD(cr, active) << "]";
    return os;
}

class Context {
    const pcl::visualization::PCLVisualizer::Ptr m_viewer;
    std::forward_list<cloud_reg> m_clouds;

  public:
    Context(pcl::visualization::PCLVisualizer::Ptr &viewer)
        : m_viewer(viewer), m_clouds(){};
    void addCloud(cloud_reg &reg);
    void handleKeyboardEvent(const pcl::visualization::KeyboardEvent &event);
    void updateInViewer(cloud_reg &cr);
};

void Context::updateInViewer(cloud_reg &cr) {
    m_viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_OPACITY, cr.active ? 1.0 : 0.0,
        cr.name);
}

void Context::addCloud(cloud_reg &reg) {
    // FSG_LOG_MSG("Adding cloud with key " << reg.key << ", name " <<
    // reg.name);
    FSG_LOG_MSG("Adding cloud " << reg);

    m_viewer->addPointCloud<pcl::PointXYZRGB>(reg.cloud, reg.name);

    m_clouds.push_front(reg);
    updateInViewer(reg);
    //    cloud_reg *newreg = &m_clouds.front();
}

void Context::handleKeyboardEvent(
    const pcl::visualization::KeyboardEvent &event) {
    if (event.keyUp()) {
        const std::string keySym = event.getKeySym();
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM("Handle key pressed '" << keySym
                                                                 << "'");

        if (keySym.compare("twosuperior") == 0) {
            FSG_LOG_MSG("Resetting camera.");
            m_viewer->setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0);
        }

        for (auto &cr : m_clouds) {
            // FSG_LOG_MSG("Checking key " << cr.key << ", name " << cr.name);
            if ((keySym.compare(cr.key) == 0)) {
                // FSG_LOG_MSG("keysym " << keySym << " matches cloud name "
                //                       << cr.name << " => toggling (was "
                //                       << cr.active << ")");

                cr.active = !cr.active;
                updateInViewer(cr);
                FSG_LOG_VAR(cr);
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

Eigen::ComputationInfo minimizationResultToComputationInfo(
    Eigen::LevenbergMarquardtSpace::Status minimizationResult) {
    switch (minimizationResult) {
    case Eigen::LevenbergMarquardtSpace::Status::NotStarted:
        return Eigen::ComputationInfo::InvalidInput;
    case Eigen::LevenbergMarquardtSpace::Status::Running:
        return Eigen::ComputationInfo::InvalidInput;
    case Eigen::LevenbergMarquardtSpace::Status::ImproperInputParameters:
        return Eigen::ComputationInfo::NumericalIssue;
    case Eigen::LevenbergMarquardtSpace::Status::RelativeReductionTooSmall:
        return Eigen::ComputationInfo::Success;
    case Eigen::LevenbergMarquardtSpace::Status::RelativeErrorTooSmall:
        return Eigen::ComputationInfo::Success;
    case Eigen::LevenbergMarquardtSpace::Status::
        RelativeErrorAndReductionTooSmall:
        return Eigen::ComputationInfo::Success;
    case Eigen::LevenbergMarquardtSpace::Status::CosinusTooSmall:
        return Eigen::ComputationInfo::Success;
    case Eigen::LevenbergMarquardtSpace::Status::TooManyFunctionEvaluation:
        return Eigen::ComputationInfo::NoConvergence;
    case Eigen::LevenbergMarquardtSpace::Status::FtolTooSmall:
        return Eigen::ComputationInfo::Success;
    case Eigen::LevenbergMarquardtSpace::Status::XtolTooSmall:
        return Eigen::ComputationInfo::Success;
    case Eigen::LevenbergMarquardtSpace::Status::GtolTooSmall:
        return Eigen::ComputationInfo::Success;
    case Eigen::LevenbergMarquardtSpace::Status::UserAsked:
        return Eigen::ComputationInfo::InvalidInput;
    }
    return Eigen::ComputationInfo::InvalidInput; // Make compiler happy.
}

void FloatTest() {
    FSG_TRACE_THIS_FUNCTION();
    float ref = 1.0;
    float epsilon = ref;
    float ref_plus_epsilon = 0;
    do {
        epsilon = epsilon / 2.0;
        // FSG_LOG_VAR(epsilon);
        ref_plus_epsilon = ref + epsilon;
        // FSG_LOG_VAR(ref_plus_epsilon);
    } while (ref_plus_epsilon != ref);
    FSG_LOG_MSG("First epsilon that added to "
                << ref << " does not change a bit: " << epsilon);
}

static const float fit_control_epsilon = 0.01;

/**
   This method generates a number of varied parameter sets (trying
   first vanilla parameters then changing each dimensions in turn),
   turns them into point clouds, and checks that the generated point
   cloud fits precisely the parameter.
 */
void SuperEllipsoidTestEachDimensionForMisbehavior(
    fsg::SuperEllipsoidParameters &superellipsoidparameters_prototype) {
    FSG_TRACE_THIS_FUNCTION();

    FSG_LOG_VAR(superellipsoidparameters_prototype);

    for (int dimension_shift = -1; dimension_shift < 11; dimension_shift++) {
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
            "SuperEllipsoidTestEachDimensionForMisbehavior dimension shift "
            << dimension_shift);
        fsg::SuperEllipsoidParameters superellipsoidparameters =
            superellipsoidparameters_prototype;

        if (dimension_shift >= 0) {
            superellipsoidparameters.coeff(dimension_shift) = 1.5;
        }

        const pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud =
            superellipsoidparameters.toPointCloud(10);

        std::vector<int> indices(pointCloud->size());
        for (size_t i = 0; i < pointCloud->size(); ++i) {
            indices[i] = i;
        }

        OptimizationFunctor functor(*pointCloud, indices);

        Eigen::VectorXf deviation(pointCloud->size());

        functor(superellipsoidparameters.coeff, deviation);

        // FSG_LOG_VAR(deviation);
        FSG_LOG_VAR(deviation.norm());

        if (deviation.norm() > fit_control_epsilon) {
            FSG_LOG_MSG("Test FAIL on dimension " << dimension_shift << ".");
        }
    }
}

void SuperEllipsoidTestComputeGradient(
    fsg::SuperEllipsoidParameters &superellipsoidparameters_prototype,
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud) {

    FSG_LOG_VAR(superellipsoidparameters_prototype);

    std::vector<int> indices(pointCloud->size());
    for (size_t i = 0; i < pointCloud->size(); ++i) {
        indices[i] = i;
    }

    OptimizationFunctor functor(*pointCloud, indices);

    Eigen::VectorXf deviation(pointCloud->size());

    functor(superellipsoidparameters_prototype.coeff, deviation);

    float center_value = deviation.norm();

    FSG_LOG_VAR(center_value);

    const float epsilon = 0.05;

    for (int dimension_shift = 0; dimension_shift < 11; dimension_shift++) {
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
            "SuperEllipsoidTestComputeGradient dimension shift "
            << dimension_shift);
        fsg::SuperEllipsoidParameters superellipsoidparameters =
            superellipsoidparameters_prototype;

        superellipsoidparameters.coeff(dimension_shift) =
            superellipsoidparameters_prototype.coeff(dimension_shift) - epsilon;
        functor(superellipsoidparameters.coeff, deviation);
        float minus = deviation.norm();

        superellipsoidparameters.coeff(dimension_shift) =
            superellipsoidparameters_prototype.coeff(dimension_shift) + epsilon;
        functor(superellipsoidparameters.coeff, deviation);
        float plus = deviation.norm();

        FSG_LOG_MSG("dimension " << dimension_shift << " values " << minus << ""
                                 << center_value << "" << plus);
    }
}

bool pointCloudToFittingContextWithInitialEstimate_EigenLevenbergMarquardt(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz,
    fsg::SuperEllipsoidParameters &fittingContext) {
    FSG_TRACE_THIS_FUNCTION();
    fsg::SuperEllipsoidParameters initialEstimate = fittingContext;

    FSG_LOG_MSG("Initial estimate : " << initialEstimate);

    std::vector<int> indices(cloud_xyz->size());
    for (size_t i = 0; i < cloud_xyz->size(); ++i) {
        indices[i] = i;
    }

    OptimizationFunctor functor(*cloud_xyz, indices);
    Eigen::NumericalDiff<OptimizationFunctor> num_diff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<OptimizationFunctor>, float>
        lm(num_diff);
    Eigen::LevenbergMarquardtSpace::Status minimizationResult;

    {
        FSG_TRACE_THIS_SCOPE_WITH_STATIC_STRING(
            "Eigen::LevenbergMarquardt::minimize()");
        Eigen::VectorXd coeff_d = fittingContext.coeff;
        Eigen::VectorXf coeff_f = coeff_d.cast<float>();
        Eigen::VectorXf foo;

        minimizationResult =
            (Eigen::LevenbergMarquardtSpace::Status)0; // lm.minimize(coeff_f);
    }

    Eigen::ComputationInfo ci =
        minimizationResultToComputationInfo(minimizationResult);

    FSG_LOG_MSG("Minimization result: Eigen::ComputationInfo="
                << ci
                << " LevenbergMarquardtSpace=" << (int)minimizationResult);

    FSG_LOG_MSG("Initial estimation : " << initialEstimate);
    FSG_LOG_MSG("After minimization : " << fittingContext);

    if (ci != Eigen::ComputationInfo::Success) {
        FSG_LOG_MSG("Levenberg-Marquardt fitting failed, with code: " << ci);
        return false;
    }
    return true;
}

bool pointCloudToFittingContextWithInitialEstimate_LibCmaes(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz,
    fsg::SuperEllipsoidParameters &fittingContext) {
    FSG_TRACE_THIS_FUNCTION();
    fsg::SuperEllipsoidParameters initialEstimate = fittingContext;

    FSG_LOG_MSG("Initial estimate : " << initialEstimate);

    std::vector<int> indices(cloud_xyz->size());
    for (size_t i = 0; i < cloud_xyz->size(); ++i) {
        indices[i] = i;
    }

    std::vector<double> x0(fsg::SuperEllipsoidParameters::fieldCount, 0);

    // copy to

    OptimizationFunctor functor(*cloud_xyz, indices);

    libcmaes::FitFunc cmaes_fit_func = [&functor, &cloud_xyz](const double *x,
                                                              const int N) {
        fsg::SuperEllipsoidParameters cmaes_eval_params;

        // FSG_LOG_VAR(N);

        if (N != fsg::SuperEllipsoidParameters::fieldCount) {
            FSG_LOG_MSG("Mismatch field/vector count: "
                        << fsg::SuperEllipsoidParameters::fieldCount << " vs. "
                        << N);
            exit(1);
        }

        for (int i = 0; i < N; i++) {
            cmaes_eval_params.coeff(i) = x[i];
        }

        // FSG_LOG_VAR(cmaes_eval_params);

        Eigen::VectorXf deviation(cloud_xyz->size());

        functor(cmaes_eval_params.coeff, deviation);

        double deviation_sum_of_squares = deviation.norm();

        // FSG_LOG_VAR(deviation_sum_of_squares);

        return deviation_sum_of_squares;
    };

    Eigen::VectorXd OptimizationStepSize(
        fsg::SuperEllipsoidParameters::fieldCount);
    OptimizationStepSize << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 3, 1.5, 1.5, 1, 1;

    // FSG_LOG_VAR(OptimizationStepSize);

    Eigen::VectorXd LowerBounds(fsg::SuperEllipsoidParameters::fieldCount);
    LowerBounds << -3, -3, -3, 0, 0, 0, -M_PI, -M_PI_2, -M_PI_2, 0.1, 0.1;

    // FSG_LOG_VAR(LowerBounds);

    Eigen::VectorXd UpperBounds(fsg::SuperEllipsoidParameters::fieldCount);
    UpperBounds << 3, 3, 3, 1, 1, 1, M_PI, M_PI_2, M_PI_2, 2, 2;

    // FSG_LOG_VAR(UpperBounds);

    // https://github.com/beniz/libcmaes/wiki/Defining-and-using-bounds-on-parameters
    libcmaes::GenoPheno<libcmaes::pwqBoundStrategy> gp(
        LowerBounds.data(), UpperBounds.data(),
        fsg::SuperEllipsoidParameters::fieldCount); // genotype / phenotype
                                                    // transform associated to
                                                    // bounds.

    libcmaes::CMAParameters<libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>>
        cmaparams(fsg::SuperEllipsoidParameters::fieldCount,
                  fittingContext.coeffData(),
                  0 /* OptimizationStepSize.data() */, -1, 0, gp);
    // CMAParameters(const dVec &x0,
    //               const dVec &sigma,
    //               const int &lambda,
    //               const dVec &lbounds,
    //               const dVec &ubounds,
    //               const uint64_t &seed);

    {
        static int n = 0;
        auto libcmaes_graph_data_filename_stringstream = std::stringstream();
        libcmaes_graph_data_filename_stringstream << "libcmaes_log_" << n << ".dat";
        std::string libcmaes_graph_data_filename = libcmaes_graph_data_filename_stringstream.str();
        FSG_LOG_VAR(libcmaes_graph_data_filename);
        cmaparams.set_fplot(libcmaes_graph_data_filename);
        n++;
    }
    
    cmaparams.set_mt_feval(true); // activates the parallel evaluation
    cmaparams.set_fixed_p(10, 1);
    cmaparams.set_fixed_p(9, 1);
    cmaparams.set_ftarget(0);

    libcmaes::CMASolutions cmasols =
        libcmaes::cmaes<libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>>(
            cmaes_fit_func, cmaparams);

    FSG_LOG_MSG("best solution: " << cmasols);
    FSG_LOG_MSG("optimization took " << cmasols.elapsed_time() / 1000.0
                                     << " seconds\n");

    int cma_status = cmasols.run_status();

    FSG_LOG_VAR(cma_status); // the optimization status, failed if < 0
    FSG_LOG_VAR(cmasols.status_msg());

    // https://github.com/beniz/libcmaes/wiki/Optimizing-a-function#user-content-solution-error-covariance-matrix-and-expected-distance-to-the-minimum-edm
    libcmaes::Candidate bcand = cmasols.best_candidate();

    FSG_LOG_BEGIN() << FSG_LOCATION() << FSG_INDENTATION() << "cmasols.print(..., 0, cmaparams.get_gp()): ";

    cmasols.print(FSG_LOG_BEGIN(),false,cmaparams.get_gp());

    FSG_LOG_BEGIN() << FSG_LOG_END();

    FSG_LOG_BEGIN() << FSG_LOCATION() << FSG_INDENTATION() << "cmasols.print(..., 1, cmaparams.get_gp()): ";

    cmasols.print(FSG_LOG_BEGIN(),false,cmaparams.get_gp());

    FSG_LOG_BEGIN() << FSG_LOG_END();


    // double fmin = bcand.get_fvalue(); // min objective function value the
    // optimizer converged to
    Eigen::VectorXd bestparameters = bcand.get_x_dvec(); // vector of objective
                                                         // function parameters
                                                         // at minimum.
    // double edm = cmasols.edm(); // expected distance to the minimum.

    // https://github.com/beniz/libcmaes/wiki/Defining-and-using-bounds-on-parameters#user-content-retrieving-the-best-solution
    // Eigen::VectorXd bestparameters =
    // gp.pheno(cmasols.get_best_seen_candidate().get_x_dvec());

    fittingContext.coeff = bestparameters;

    FSG_LOG_MSG("Initial estimation : " << initialEstimate);
    FSG_LOG_MSG("After minimization : " << fittingContext);

    {
        /* Check  */
        Eigen::VectorXf deviation(cloud_xyz->size());
        functor(fittingContext.coeff, deviation);
        // FSG_LOG_VAR(deviation);
        FSG_LOG_VAR(deviation.norm());
    }

    if (cma_status < 0) {
        FSG_LOG_MSG("CMAES fitting failed, with code: " << cma_status);
        return false;
    }
    return true;
}

void SuperEllipsoidComputeGradientAllDimensions(
    fsg::SuperEllipsoidParameters &superellipsoidparameters_center,
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud,
    fsg::SuperEllipsoidParameters *gradient) {
    FSG_TRACE_THIS_FUNCTION();

    FSG_LOG_VAR(superellipsoidparameters_center);

    std::vector<int> indices(pointCloud->size());
    for (size_t i = 0; i < pointCloud->size(); ++i) {
        indices[i] = i;
    }

    OptimizationFunctor functor(*pointCloud, indices);

    Eigen::VectorXf deviation(pointCloud->size());

    Eigen::VectorXf values(5);
    const float epsilon = 0.001;

    for (int dimension_shift = 0; dimension_shift < 11; dimension_shift++) {
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
            "SuperEllipsoidComputeGradientAllDimensions dimension shift "
            << dimension_shift);

        for (int step = -1; step <= 1; step += 1) {
            float stepEpsilon = step * epsilon;

            fsg::SuperEllipsoidParameters superellipsoidparameters =
                superellipsoidparameters_center;

            superellipsoidparameters.coeff(dimension_shift) += epsilon * step;

            FSG_LOG_VAR(stepEpsilon);
            FSG_LOG_VAR(superellipsoidparameters);

            functor(superellipsoidparameters.coeff, deviation);

            // FSG_LOG_VAR(deviation);
            float value = deviation.norm();
            FSG_LOG_VAR(value);

            values[step + 1] = value;
        }

        gradient->coeff(dimension_shift) = values[2] - values[0];
    }
    FSG_LOG_VAR(*gradient);
}

bool pointCloudToFittingContextWithInitialEstimate_both(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz,
    fsg::SuperEllipsoidParameters &fittingContext) {

    fsg::SuperEllipsoidParameters gradient_wrt_pointcloud;
    SuperEllipsoidComputeGradientAllDimensions(fittingContext, cloud_xyz,
                                               &gradient_wrt_pointcloud);

    fsg::SuperEllipsoidParameters fittingContext_eigenlevenbergmarquardt(
        fittingContext);
    bool success_eigenlevenbergmarquardt =
        pointCloudToFittingContextWithInitialEstimate_EigenLevenbergMarquardt(
            cloud_xyz, fittingContext_eigenlevenbergmarquardt);
    fsg::SuperEllipsoidParameters fittingContext_libcmaes(fittingContext);
    bool success_libcmaes =
        pointCloudToFittingContextWithInitialEstimate_LibCmaes(
            cloud_xyz, fittingContext_libcmaes);
    FSG_LOG_MSG("pointCloudToFittingContextWithInitialEstimate_both results: "
                "EigenLevenbergMarquardt="
                << (success_eigenlevenbergmarquardt ? "success" : "failure")
                << ", libcmaes result="
                << (success_libcmaes ? "success" : "failure"));

    if (success_eigenlevenbergmarquardt) {
        fittingContext = fittingContext_eigenlevenbergmarquardt;
        FSG_LOG_MSG("pointCloudToFittingContextWithInitialEstimate_both: "
                    "returning EigenLevenbergMarquardt result");
        return true;
    }

    if (success_libcmaes) {
        fittingContext = fittingContext_libcmaes;
        FSG_LOG_MSG("pointCloudToFittingContextWithInitialEstimate_both: "
                    "returning libcmaes result");
        return true;
    }

    FSG_LOG_MSG("pointCloudToFittingContextWithInitialEstimate_both: both "
                "failed.  Returning initial estimation.");
    return false;
}

bool pointCloudToFittingContextWithInitialEstimate(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz,
    fsg::SuperEllipsoidParameters &fittingContext) {
    return pointCloudToFittingContextWithInitialEstimate_both(cloud_xyz,
                                                              fittingContext);
}

/**
    Given a parameter set and a point cloud which exactly matches,
    verify that disturbing in any dimension yields a sane gradient
    that converges back to the original parameter set.
*/
void SuperEllipsoidTestEachDimensionForGradientSanity(
    fsg::SuperEllipsoidParameters &superellipsoidparameters_center,
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud) {
    FSG_TRACE_THIS_FUNCTION();

    FSG_LOG_VAR(superellipsoidparameters_center);

    std::vector<int> indices(pointCloud->size());
    for (size_t i = 0; i < pointCloud->size(); ++i) {
        indices[i] = i;
    }

    OptimizationFunctor functor(*pointCloud, indices);

    Eigen::VectorXf deviation(pointCloud->size());

    Eigen::VectorXf values(5);
    const float epsilon = 0.001;

    functor(superellipsoidparameters_center.coeff, deviation);
    float centervalue = deviation.norm();
    FSG_LOG_VAR(centervalue);
    values[2] = centervalue;

    for (int dimension_shift = 0; dimension_shift < 11; dimension_shift++) {
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
            "SuperEllipsoidTestEachDimensionForGradientSanity dimension shift "
            << dimension_shift);

        /* First check that gradient itself is good. */
        for (int step = -2; step <= 2; step++) {
            if (step == 0) {
                continue;
            }
            float stepEpsilon = step * epsilon;

            fsg::SuperEllipsoidParameters superellipsoidparameters =
                superellipsoidparameters_center;

            superellipsoidparameters.coeff(dimension_shift) += epsilon * step;

            // FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
            //     "dimension shift " << dimension_shift << " stepEpsilon "
            //                        << stepEpsilon);
            FSG_LOG_VAR(stepEpsilon);
            FSG_LOG_VAR(superellipsoidparameters);

            functor(superellipsoidparameters.coeff, deviation);

            // FSG_LOG_VAR(deviation);
            float value = deviation.norm();
            FSG_LOG_VAR(value);

            values[step + 2] = value;
        }

        FSG_LOG_VAR(values);
        FSG_LOG_MSG("values=[" << values[0] << ", " << values[1] << ",  "
                               << values[2] << "  , " << values[3] << ", "
                               << values[4] << "]");

        {
            int failures = 0;
            if (values[0] < values[1]) {
                FSG_LOG_MSG("FAIL: bad gradient lower side on dimension "
                            << dimension_shift << " values " << values
                            << " params " << superellipsoidparameters_center);
                failures++;
            }

            if (values[3] > values[4]) {
                FSG_LOG_MSG("FAIL: bad gradient higher side on dimension "
                            << dimension_shift << " values " << values
                            << " params " << superellipsoidparameters_center);
                failures++;
            }

            if (values[2] > values[1]) {
                FSG_LOG_MSG(
                    "FAIL: bad gradient center higher than left on dimension "
                    << dimension_shift << " values " << values << " params "
                    << superellipsoidparameters_center);
                failures++;
            }

            if (values[2] > values[3]) {
                FSG_LOG_MSG(
                    "FAIL: bad gradient center higher than right on dimension "
                    << dimension_shift << " values " << values << " params "
                    << superellipsoidparameters_center);
                failures++;
            }

            if (failures == 0) {
                FSG_LOG_MSG("Good gradient on dimension "
                            << dimension_shift << " params "
                            << superellipsoidparameters_center << " values "
                            << values);
            }
        }

        /** Assuming that the gradient is good, we provide to the
         * optimizer the actual point cloud and an estimate which is
         * perfect for all dimensions except the one we just tested
         * the gradient on.  Unless the optimizer is *really* broken,
         * it should easily find the parameters.
         */

        fsg::SuperEllipsoidParameters superellipsoidparameters_fit =
            superellipsoidparameters_center;

        superellipsoidparameters_fit.coeff(dimension_shift) += epsilon;

        /* execute the optimization */
        bool success = pointCloudToFittingContextWithInitialEstimate(
            pointCloud, superellipsoidparameters_fit);

        if (!success) {
            FSG_LOG_MSG("Fit after gradient epsilon failed, thus gradient test "
                        "FAIL, on dimension "
                        << dimension_shift);
            continue;
        }

        FSG_LOG_MSG("Fit after gradient epsilon converges, on dimension "
                    << dimension_shift);
        FSG_LOG_VAR(superellipsoidparameters_fit);
        FSG_LOG_VAR(superellipsoidparameters_center);

        Eigen::VectorXd deviation = superellipsoidparameters_fit.coeff -
                                    superellipsoidparameters_center.coeff;

        FSG_LOG_VAR(deviation.norm());

        if (deviation.norm() > fit_control_epsilon) {
            FSG_LOG_MSG(
                "Test FAIL fitting parameter at epsilon deviation on dimension "
                << dimension_shift);
        }
    }
}

bool pointCloudToFittingContext(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz,
    fsg::SuperEllipsoidParameters &fittingContext,
    pcl::visualization::PCLVisualizer *viewer,
    const std::string &obj_index_i_s) {
    FSG_TRACE_THIS_FUNCTION();
    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(cloud_xyz);
    // Minimize eccentricity computation.
    feature_extractor.setAngleStep(360);
    feature_extractor.compute();

    Eigen::Vector3f mass_center;
    feature_extractor.getMassCenter(
        mass_center); // FIXME should check return value

    FSG_LOG_VAR(mass_center);

    Eigen::Vector3f major_vector, middle_vector, minor_vector;

    feature_extractor.getEigenVectors(major_vector, middle_vector,
                                      minor_vector);
    FSG_LOG_VAR(major_vector);
    FSG_LOG_VAR(middle_vector);
    FSG_LOG_VAR(minor_vector);

    pcl::PointXYZ min_point_OBB;
    pcl::PointXYZ max_point_OBB;
    pcl::PointXYZ position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    feature_extractor.getOBB(
        min_point_OBB, max_point_OBB, position_OBB,
        rotational_matrix_OBB); // FIXME should check return value

    FSG_LOG_VAR(position_OBB);
    FSG_LOG_VAR(min_point_OBB);
    FSG_LOG_VAR(max_point_OBB);

    FSG_LOG_VAR(rotational_matrix_OBB);

    // FIXME clarify/generalize major/z.
    major_vector *= (max_point_OBB.z - min_point_OBB.z) / 2.0;
    middle_vector *= (max_point_OBB.y - min_point_OBB.y) / 2.0;
    minor_vector *= (max_point_OBB.x - min_point_OBB.x) / 2.0;
    FSG_LOG_VAR(major_vector);
    FSG_LOG_VAR(middle_vector);
    FSG_LOG_VAR(minor_vector);

    // From
    // http://pointclouds.org/documentation/tutorials/moment_of_inertia.php

    Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
    Eigen::Quaternionf quat(rotational_matrix_OBB);

    pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));

    // FIXME clarify/generalize major/z.
    pcl::PointXYZ maj_axis(2.0 * major_vector(0) + mass_center(0),
                           2.0 * major_vector(1) + mass_center(1),
                           2.0 * major_vector(2) + mass_center(2));
    pcl::PointXYZ mid_axis(2.0 * middle_vector(0) + mass_center(0),
                           2.0 * middle_vector(1) + mass_center(1),
                           2.0 * middle_vector(2) + mass_center(2));
    pcl::PointXYZ min_axis(2.0 * minor_vector(0) + mass_center(0),
                           2.0 * minor_vector(1) + mass_center(1),
                           2.0 * minor_vector(2) + mass_center(2));

    if (viewer != NULL) {
        std::string obbId(obj_index_i_s + ":obb");
        FSG_LOG_MSG("will add obb with id: " << obbId);

        viewer->addCube(position, quat, max_point_OBB.x - min_point_OBB.x,
                        max_point_OBB.y - min_point_OBB.y,
                        max_point_OBB.z - min_point_OBB.z, obbId);

        viewer->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, obbId);

        viewer->addLine(center, maj_axis, 1.0f, 0.0f, 0.0f,
                        obj_index_i_s + ":major eigen vector");
        viewer->addLine(center, mid_axis, 0.0f, 1.0f, 0.0f,
                        obj_index_i_s + ":middle eigen vector");
        viewer->addLine(center, min_axis, 0.0f, 0.0f, 1.0f,
                        obj_index_i_s + ":minor eigen vector");
    }

    fittingContext.set_cen_x(mass_center(0));
    fittingContext.set_cen_y(mass_center(1));
    fittingContext.set_cen_z(mass_center(2));

    fittingContext.set_rad_a(major_vector.norm());

    fittingContext.set_rad_b(middle_vector.norm());

    fittingContext.set_rad_c(minor_vector.norm());

    float yaw, pitch, roll;
    matrix_to_angles(rotational_matrix_OBB, yaw, pitch, roll);

    FSG_LOG_MSG("yaw=" << yaw << ", pitch=" << pitch << ", roll=" << roll);

    fittingContext.set_rot_yaw(yaw);
    fittingContext.set_rot_pitch(pitch);
    fittingContext.set_rot_roll(roll);

    fittingContext.set_exp_1(1);
    fittingContext.set_exp_2(1);

    return pointCloudToFittingContextWithInitialEstimate(cloud_xyz,
                                                         fittingContext);
}

bool SuperEllipsoidFitARandomSQ(boost::random::minstd_rand &_gen) {
    FSG_TRACE_THIS_FUNCTION();
    boost::random::uniform_real_distribution<> random_float_m5p5(-1, 1);
    boost::random::uniform_real_distribution<> random_float_cent_one(0.01, 1);
    boost::random::uniform_real_distribution<> random_float_mpippi(-M_PI, M_PI);
    // ost::random::uniform_real_distribution<> random_float_cent_two(1, 1);

    fsg::SuperEllipsoidParameters sep_groundtruth;
    sep_groundtruth.set_cen_x(random_float_m5p5(_gen));
    sep_groundtruth.set_cen_y(random_float_m5p5(_gen));
    sep_groundtruth.set_cen_z(random_float_m5p5(_gen));
    sep_groundtruth.set_rad_a(random_float_cent_one(_gen));
    sep_groundtruth.set_rad_b(random_float_cent_one(_gen));
    sep_groundtruth.set_rad_c(random_float_cent_one(_gen));
    sep_groundtruth.set_rot_yaw(random_float_mpippi(_gen));
    sep_groundtruth.set_rot_pitch(random_float_mpippi(_gen));
    sep_groundtruth.set_rot_roll(random_float_mpippi(_gen));
    sep_groundtruth.set_exp_1(1); // random_float_cent_two(_gen));
    sep_groundtruth.set_exp_2(1); // random_float_cent_two(_gen));

    FSG_LOG_VAR(sep_groundtruth);

    SuperEllipsoidTestEachDimensionForMisbehavior(sep_groundtruth);

    FSG_LOG_VAR(sep_groundtruth);

    const pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud =
        sep_groundtruth.toPointCloud(4);

    SuperEllipsoidTestEachDimensionForGradientSanity(sep_groundtruth,
                                                     pointCloud);

    FSG_LOG_MSG("Now testing actual fit.");
    FSG_LOG_VAR(sep_groundtruth);

    fsg::SuperEllipsoidParameters sep_fit;

    bool success = pointCloudToFittingContext(pointCloud, sep_fit, nullptr, "");

    if (!success) {
        FSG_LOG_MSG(
            "Fit failed, thus test FAIL, on parameter: " << sep_groundtruth);
        return false;
    }

    FSG_LOG_VAR(sep_groundtruth);
    FSG_LOG_VAR(sep_fit);

    Eigen::VectorXd deviation = sep_fit.coeff - sep_groundtruth.coeff;

    FSG_LOG_VAR(deviation.norm());

    if (deviation.norm() > fit_control_epsilon) {
        FSG_LOG_MSG("Test FAIL fitting parameter: " << sep_groundtruth);
        return false;
    }
    FSG_LOG_MSG("Test success fitting parameter: " << sep_groundtruth);
    return true;
}

void SuperEllipsoidTest() {
    {
        fsg::SuperEllipsoidParameters superellipsoidparameters_prototype =
            fsg::SuperEllipsoidParameters::Default();
        SuperEllipsoidTestEachDimensionForMisbehavior(
            superellipsoidparameters_prototype);
    }

    {
        boost::random::minstd_rand _gen;
        FSG_TRACE_THIS_SCOPE_WITH_STATIC_STRING(
            "100 times SuperEllipsoidFitARandomSQ");
        for (int i = 0; i < 100; i++) {
            FSG_TRACE_THIS_SCOPE_WITH_SSTREAM("Fitting random " << i);
            SuperEllipsoidFitARandomSQ(_gen);
        }
    }
}

int main(int argc, char **argv) {
    FSG_LOG_INIT__CALL_FROM_CPP_MAIN();
    FSG_TRACE_THIS_FUNCTION();

    if (argc == 2) {
        std::string test = "test";
        if (test.compare(argv[1]) == 0) {
            FloatTest();
            SuperEllipsoidTest();
            return 0;
        }
    }

    if (argc != 4) {

        std::cerr << "Usage : \n\t- pcd file\n\t- gmm archive\n\t- label"
                  << std::endl;
        std::cerr << "To run self-test : \n\ttest" << std::endl;
        return 1;
    }

    FSG_LOG_VAR(fit_control_epsilon);

    std::string gmm_archive = argv[2];
    std::string label = argv[3];

    //* Load pcd file into a pointcloud
    ip::PointCloudT::Ptr input_cloud_soi(new ip::PointCloudT);

    {
        std::string pcd_file = argv[1];
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM("load pcd file: " << pcd_file);
        pcl::io::loadPCDFile(pcd_file, *input_cloud_soi);
    }

    ip::SurfaceOfInterest soi(input_cloud_soi);

    {
        FSG_TRACE_THIS_SCOPE_WITH_STATIC_STRING("handle gmm");
        iagmm::GMM gmm;

        {
            FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
                "load classifier archive: " << gmm_archive);
            //* Load the CMMs classifier from the archive
            std::ifstream ifs(gmm_archive);
            if (!ifs) {
                FSG_LOG_MSG("Unable to open archive : " << gmm_archive);
                return 1;
            }
            boost::archive::text_iarchive iarch(ifs);
            iarch >> gmm;
            //*/
        }

        //* Generate relevance map on the pointcloud
        {
            FSG_TRACE_THIS_SCOPE_WITH_STATIC_STRING("compute supervoxels");
            soi.computeSupervoxel();
            FSG_LOG_MSG(soi.getSupervoxels().size()
                        << " supervoxels extracted");
        }

        {
            FSG_TRACE_THIS_SCOPE_WITH_STATIC_STRING("compute meanFPFHLabHist");
            soi.compute_feature("meanFPFHLabHist");
        }

        {
            FSG_TRACE_THIS_SCOPE_WITH_STATIC_STRING(
                "compute meanFPFHLabHist weights");
            soi.compute_weights<iagmm::GMM>("meanFPFHLabHist", gmm);
        }
    }

    //* Generate objects hypothesis
    std::vector<std::set<uint32_t>> obj_hypotheses;
    {
        FSG_TRACE_THIS_SCOPE_WITH_STATIC_STRING("soi.extract_regions");
        obj_hypotheses = soi.extract_regions("meanFPFHLabHist", 0.5, 1);
        FSG_LOG_MSG(obj_hypotheses.size() << " objects hypothesis extracted");
    }
    //*/

    // obj_hypotheses

    std::string windowTitle;

    {
        std::stringstream ss;
        ss << "Object fit viewer : " << label;
        windowTitle = ss.str();
    }

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer(label));
    boost::shared_ptr<Context> context_p(new Context(viewer));

    {
        FSG_TRACE_THIS_SCOPE_WITH_STATIC_STRING("Computations");
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
                    // FSG_LOG_MSG( " skipping sv of label " << current_sv_label
                    // <<
                    // " weight " << c );
                    continue;
                }
                // FSG_LOG_MSG( " KEEPING sv of label " << current_sv_label << "
                // weight " << c );
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

            FSG_LOG_MSG("Thresholding kept " << kept << " supervoxels out of "
                                             << supervoxels.size());

            /* Populate again with cloud fitted with shape. */

            /* We have to express what supervoxels belong together.

               We could copy points, or just set indices, which saves memory.
               Actually, PCL uses indices anyway.

               We don't have to filter again because extract_regions already
               does
               it.

            */

            // Rappel : typedef std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>
            // SupervoxelArray;

            /* each object */
            for (const auto &obj_hyp :
                 obj_hypotheses | boost::adaptors::indexed(0)) {

                std::string obj_index_i_s = std::to_string(obj_hyp.index());
                FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
                    "Considering obj hypothesis id=" << obj_index_i_s);

                std::set<uint32_t> *p_obj_hyp = &(obj_hyp.value());

                int r = float(dist(_gen) * 85);
                int g = float(dist(_gen) * 85);
                int b = float(dist(_gen) * 85);

                FSG_LOG_MSG("Assigned color = " << r << "," << g << "," << b);

                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(
                    new pcl::PointCloud<pcl::PointXYZ>);

                if (p_obj_hyp->size() <= 1) {
                    FSG_LOG_MSG("Skipping hypothesis object id="
                                << obj_index_i_s
                                << " because too few supervoxels: "
                                << p_obj_hyp->size());
                    continue;
                }

                {
                    FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
                        "Gather supervoxels into point cloud for hypothesis id="
                        << obj_index_i_s);

                    int kept = 0;
                    pcl::PointXYZ pt;
                    for (auto it_sv = supervoxels.begin();
                         it_sv != supervoxels.end(); it_sv++) {
                        int current_sv_label = it_sv->first;
                        pcl::Supervoxel<ip::PointT>::Ptr current_sv =
                            it_sv->second;

                        if (p_obj_hyp->find(current_sv_label) ==
                            p_obj_hyp->end()) {
                            // FSG_LOG_MSG( "Supervoxel " << current_sv_label <<
                            // "
                            // not part of current object, skipping." );
                            continue;
                        }
                        ++kept;

                        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
                            "Supervoxel labelled "
                            << current_sv_label
                            << " is part of current object hypothesis id="
                            << obj_index_i_s << ", including, "
                                                "will add "
                            << current_sv->voxels_->size() << " point(s).");
                        for (auto v : *(current_sv->voxels_)) {
                            pt.x = v.x;
                            pt.y = v.y;
                            pt.z = v.z;
                            cloud_xyz->push_back(pt);
                        }
                    }
                    FSG_LOG_MSG("Gathered "
                                << kept
                                << " supervoxels into a point cloud of size "
                                << cloud_xyz->size()
                                << " for hypothesis id=" << obj_index_i_s);
                }

                if (cloud_xyz->size() < 20) {
                    FSG_LOG_MSG(
                        "Skipping hypothesis object id="
                        << obj_index_i_s
                        << " because supervoxels combined into too few points: "
                        << cloud_xyz->size());
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

                fsg::SuperEllipsoidParameters fittingContext;

                bool success = pointCloudToFittingContext(
                    cloud_xyz, fittingContext, &(*viewer), obj_index_i_s);

                if (success) {

                    pcl::PointCloud<pcl::PointXYZ>::Ptr proj_points =
                        fittingContext.toPointCloud(100);

                    {
                        pcl::PointXYZRGB pt;

                        r = 127.0 + r / 2.0;
                        g = 127.0 + g / 2.0;
                        b = 127.0 + b / 2.0;
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
                } else {
                    FSG_LOG_MSG("Fitting fail on id=" << obj_index_i_s << ".");
                }

                FSG_LOG_MSG("End new obj hyp, id=" << obj_index_i_s << ".");
            }
        }

        cloud_reg clouds[] = {
            {"1", input_cloud_ptr, "input", true},
            {"2", supervoxel_cloud_ptr, "supervoxel", true},
            {"3", object_hyps_cloud_ptr, "object_hyps", true},
            {"4", superellipsoids_cloud_ptr, "superellipsoids", true},
            //{ "t", superellipsoid_cloud, "superellipsoid_cloud" },
        };

        for (auto &cr : clouds) {
            context_p->addCloud(cr);
            viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, cr.name);
        }
    }

    // viewer->addCoordinateSystem (1.0);
    viewer->setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0);

    viewer->registerKeyboardCallback(keyboardEventOccurred, (void *)&context_p);

    vtkRenderWindowInteractor *interactor =
        viewer->getRenderWindow()->GetInteractor();
    viewer->addOrientationMarkerWidgetAxes(interactor);

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    viewer->close();

    return 0;
}
