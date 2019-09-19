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
#include "git_version.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#include "cmaes.h"
#pragma GCC diagnostic pop

#include <boost/archive/text_iarchive.hpp>
#include <iagmm/gmm.hpp>
#include <math.h>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include "fsg_trace.hpp"
#include "number_type.hpp"
#include "test_rotation.hpp"

namespace ip = image_processing;

using namespace fsg::matrixrotationangles;

namespace fsg
{
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudT;
typedef fsg::PointCloudT::Ptr PointCloudTP;

void pointCloudLogSomeVariables(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud)
{
    FSG_TRACE_THIS_FUNCTION();
    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(point_cloud);
    // Minimize eccentricity computation.
    feature_extractor.setAngleStep(360);
    feature_extractor.compute();

    Eigen::Vector3f
        mass_center; // Type imposed by pcl::MomentOfInertiaEstimation
    feature_extractor.getMassCenter(
        mass_center); // FIXME should check return value

    pcl::PointXYZ min_point_OBB;
    pcl::PointXYZ max_point_OBB;
    pcl::PointXYZ position_OBB;
    Eigen::Matrix3f
        rotational_matrix_OBB; // Type imposed by pcl::MomentOfInertiaEstimation
    feature_extractor.getOBB(
        min_point_OBB, max_point_OBB, position_OBB,
        rotational_matrix_OBB); // FIXME should check return value

    FSG_LOG_VAR(mass_center);
    FSG_LOG_VAR(min_point_OBB);
    FSG_LOG_VAR(max_point_OBB);
    FSG_LOG_VAR(position_OBB);
    FSG_LOG_VAR(rotational_matrix_OBB);
}

/**
    center x,y,z,
    rotation angles yaw, pitch, roll,
    radii 1,2,3,
    exponent1, exponent 2

    Total 11 parameters
*/
struct SuperEllipsoidParameters
{
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

    static SuperEllipsoidParameters Zero()
    {
        SuperEllipsoidParameters zero;
#define FSGX(name) zero.set_##name(sg_0);
        ALL_SuperEllipsoidParameters_FIELDS;
#undef FSGX
        return zero;
    }

    static SuperEllipsoidParameters Default()
    {
        SuperEllipsoidParameters dv = Zero();

        dv.set_rad_a(1.0);
        dv.set_rad_b(2.0);
        dv.set_rad_c(3.0);
        dv.set_exp_1(1.0);
        dv.set_exp_2(1.0);
        return dv;
    }

    enum idx
    {
#define FSGX(name) name,
        ALL_SuperEllipsoidParameters_FIELDS
#undef FSGX
    };

#define FSGX(name)                                                             \
    void set_##name(FNUM_TYPE f) { coeff(idx::name) = f; };
    ALL_SuperEllipsoidParameters_FIELDS;
#undef FSGX

#define FSGX(name)                                                             \
    FNUM_TYPE get_##name() const { return coeff(idx::name); };
    ALL_SuperEllipsoidParameters_FIELDS;
#undef FSGX

    static constexpr int fieldCount = 0
#define FSGX(name) +1
        ALL_SuperEllipsoidParameters_FIELDS
#undef FSGX
        ;

    // https://stackoverflow.com/questions/11490988/c-compile-time-error-expected-identifier-before-numeric-constant
    VECTORX coeff = VECTORX(fieldCount);

    FNUM_TYPE *coeffData();

    SuperEllipsoidParameters() : coeff(fieldCount){};

    friend ostream &operator<<(ostream &os,
                               const SuperEllipsoidParameters &sefc);

    pcl::PointCloud<pcl::PointXYZ>::Ptr toPointCloud(int steps);
};

constexpr int SuperEllipsoidParameters::fieldCount;

ostream &operator<<(ostream &os, const SuperEllipsoidParameters &sefc)
{
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

SuperEllipsoidParameters operator*(FNUM_TYPE d,
                                   const SuperEllipsoidParameters &sefc)
{
    SuperEllipsoidParameters result;
#define FSGX(name) result.set_##name(d *sefc.get_##name());
    ALL_SuperEllipsoidParameters_FIELDS;
#undef FSGX
    return result;
}

SuperEllipsoidParameters operator+(const SuperEllipsoidParameters &sep1,
                                   const SuperEllipsoidParameters &sep2)
{
    SuperEllipsoidParameters result;
#define FSGX(name) result.set_##name(sep1.get_##name() + sep2.get_##name());
    ALL_SuperEllipsoidParameters_FIELDS;
#undef FSGX
    return result;
}

FNUM_TYPE sym_pow(FNUM_TYPE x, FNUM_TYPE y)
{
    if (std::signbit(x) != 0)
        return -WITH_SUFFIX_fx(pow)(-x, y);
    else
        return WITH_SUFFIX_fx(pow)(x, y);
}

/** Provide access to coeff data as C-style array.  Number of
    values is provided in SuperEllipsoidParameters::fieldcount.
 */
FNUM_TYPE *SuperEllipsoidParameters::coeffData()
{
    return (this->coeff.data());
}

/** This method implements the forward transformation from a
    12-dimension model to a point cloud.

    Compare with the inverse transformation implemented in `struct
    OptimizationFunctor`.
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr
SuperEllipsoidParameters::toPointCloud(int steps)
{
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

    FNUM_TYPE exp_1 = this->get_exp_1();
    FNUM_TYPE exp_2 = this->get_exp_2();

    FNUM_TYPE dilatfactor_x = this->get_rad_a();
    FNUM_TYPE dilatfactor_y = this->get_rad_b();
    FNUM_TYPE dilatfactor_z = this->get_rad_c();

    pcl::PointXYZ pt;
    const FNUM_TYPE increment = sg_pi_2 / (FNUM_TYPE)steps;

    // Pitch is eta in Biegelbauer et al.
    for (FNUM_TYPE pitch = -sg_pi_2; pitch < sg_pi_2; pitch += increment)
    {
        FNUM_TYPE z =
            dilatfactor_z * sym_pow(WITH_SUFFIX_fx(sin)(pitch), exp_1);
        FNUM_TYPE cos_pitch_exp_1 = sym_pow(WITH_SUFFIX_fx(cos)(pitch), exp_1);

        // Yaw is omega in Biegelbauer et al.
        for (FNUM_TYPE yaw = -sg_pi; yaw < sg_pi; yaw += increment)
        {
            auto x = dilatfactor_x * sym_pow(WITH_SUFFIX_fx(cos)(yaw), exp_2) *
                     cos_pitch_exp_1;
            auto y = dilatfactor_y * sym_pow(WITH_SUFFIX_fx(sin)(yaw), exp_2) *
                     cos_pitch_exp_1;

            if ((x * x + y * y + z * z) < FNUM_LITERAL(20.0))
            {
                pt.x = (PCL_POINT_COORD_TYPE)x;
                pt.y = (PCL_POINT_COORD_TYPE)y;
                pt.z = (PCL_POINT_COORD_TYPE)z;
                cloud_step1->push_back(pt);
            }
            else
            {
                FSG_LOG_MSG("Ignoring too far point " << pt);
            }
        }
    }

    // Next rotate the point cloud.

    MATRIX3 rotmat;
    angles_to_matrix(get_rot_yaw(), get_rot_pitch(), get_rot_roll(), rotmat);

    FSG_LOG_VAR(rotmat);

    MATRIX4 transform = MATRIX4::Identity();
    transform.block(0, 0, 3, 3) << rotmat;
    // VECTOR3 center;
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
struct OptimizationFunctor : pcl::Functor<FNUM_TYPE>
{
    /** Functor constructor
     * \param[in] source cloud
     * \param[in] indices the indices of data points to evaluate
     */
    OptimizationFunctor(const pcl::PointCloud<pcl::PointXYZ> &cloud,
                        const std::vector<int> &indices)
        : pcl::Functor<FNUM_TYPE>((int)indices.size()), cloud_(cloud),
          indices_(indices)
    {
        FSG_LOG_MSG("Created functor with value count: " << values());
    }

#define pow_abs(x, y) WITH_SUFFIX_fx(pow)(WITH_SUFFIX_fx(fabs)(x), y)
// FNUM_TYPE pow_abs(const FNUM_TYPE x, const FNUM_TYPE y) const {
//     FSG_TRACE_THIS_FUNCTION();
//     FSG_LOG_VAR(x);
//     FSG_LOG_VAR(y);
//     const FNUM_TYPE absx = fabs(x);
//     FSG_LOG_VAR(absx);
//     const FNUM_TYPE result = powf(absx, y);
//     FSG_LOG_VAR(result);
//     return result;
// }

#define FUNCTOR_LOG_INSIDE 0

    /** Cost function to be minimized
     * \param[in] x the variables array
     * \param[out] fvec the resultant functions evaluations
     * \return 0
     */
    int operator()(const VECTORX &param, VECTORX &fvec) const
    {
#if FUNCTOR_LOG_INSIDE == 1
        fsg::SuperEllipsoidParameters *sep =
            (fsg::SuperEllipsoidParameters *)((void *)&param); // Yeww hack.
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM("f(): " << *sep);
#endif

        const FNUM_TYPE exp_1 =
            param(fsg::SuperEllipsoidParameters::idx::exp_1);
        // FSG_LOG_VAR(exp_1);
        const FNUM_TYPE exp_2 =
            param(fsg::SuperEllipsoidParameters::idx::exp_2);
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
        VECTOR3 cen(param(fsg::SuperEllipsoidParameters::idx::cen_x),
                    param(fsg::SuperEllipsoidParameters::idx::cen_y),
                    param(fsg::SuperEllipsoidParameters::idx::cen_z));
        // FSG_LOG_VAR(cen);

        // Compute rotation matrix
        MATRIX3 rotmat;
        angles_to_matrix(param(fsg::SuperEllipsoidParameters::idx::rot_yaw),
                         param(fsg::SuperEllipsoidParameters::idx::rot_pitch),
                         param(fsg::SuperEllipsoidParameters::idx::rot_roll),
                         rotmat);
        // FSG_LOG_VAR(rotmat);
        rotmat.transposeInPlace();
        // FSG_LOG_VAR(rotmat);

        const FNUM_TYPE two_over_exp_1 = sg_2 / exp_1;
        const FNUM_TYPE two_over_exp_2 = sg_2 / exp_2;
        const FNUM_TYPE exp_2_over_exp_1 = exp_2 / exp_1;
        // FSG_LOG_VAR(two_over_exp_2);
        // FSG_LOG_VAR(two_over_exp_1);
        // FSG_LOG_VAR(exp_2_over_exp_1);

        // FNUM_TYPE sum_of_squares = 0;

        for (signed int i = 0; i < values(); ++i)
        {
            // Take current point;
            const pcl::PointXYZ p_f = cloud_.points[indices_[i]];
            const VECTOR3 p(p_f.x, p_f.y, p_f.z);
            // FSG_LOG_VAR(p);

            // Compute vector from center.
            const VECTOR3 v_raw = p - cen;
            // FSG_LOG_VAR(v_raw);

            // Rotate vector
            const VECTOR3 v_aligned = rotmat * v_raw;
            // FSG_LOG_VAR(v_aligned);

            // TODO check major/middle/minor vs X,Y,Z...

            VECTOR3 v_scaled;
            // FIXME radii here are not major middle minor, only x y z or 1 2 3.
            v_scaled << v_aligned(0) /
                            param(fsg::SuperEllipsoidParameters::idx::rad_a),
                v_aligned(1) / param(fsg::SuperEllipsoidParameters::idx::rad_b),
                v_aligned(2) / param(fsg::SuperEllipsoidParameters::idx::rad_c);
            // FSG_LOG_VAR(v_scaled);

            const FNUM_TYPE term = pow_abs(v_scaled(0), two_over_exp_2) +
                                   pow_abs(v_scaled(1), two_over_exp_2);
            // FSG_LOG_VAR(term);

            const FNUM_TYPE outside_if_over_1 =
                pow_abs(term, exp_2_over_exp_1) +
                pow_abs(v_scaled(2), two_over_exp_1);
            // FSG_LOG_VAR(outside_if_over_1);

            const FNUM_TYPE deviation = outside_if_over_1 - 1;
#if FUNCTOR_LOG_INSIDE == 1
            FSG_LOG_VAR(deviation);
#endif

            fvec[i] = deviation;
            // sum_of_squares += deviation * deviation;
        }
        // FSG_LOG_VAR(fvec);
        //#if FUNCTOR_LOG_INSIDE == 1
        // FSG_LOG_VAR(sum_of_squares);
        //#endif
        // return (sum_of_squares); // FIXME returns int.
        return 0;
    }

    const pcl::PointCloud<pcl::PointXYZ> &cloud_;
    const std::vector<int> &indices_;
};

struct cloud_reg
{
    const char *key;
    const fsg::PointCloudTP cloud;
    const char *const name;
    bool active; // code smell: tied to a specific viewer

    friend ostream &operator<<(ostream &os, const cloud_reg &sefc);
};

ostream &operator<<(ostream &os, const cloud_reg &cr)
{
    os << "[cloud_reg " << FSG_OSTREAM_FIELD(cr, key)
       << FSG_OSTREAM_FIELD(cr, name) << FSG_OSTREAM_FIELD(cr, active) << "]";
    return os;
}

class Context
{
    const pcl::visualization::PCLVisualizer::Ptr m_viewer;
    std::forward_list<cloud_reg> m_clouds;

  public:
    Context(pcl::visualization::PCLVisualizer::Ptr &viewer)
        : m_viewer(viewer), m_clouds(){};
    void addCloud(cloud_reg &reg);
    void handleKeyboardEvent(const pcl::visualization::KeyboardEvent &event);
    void updateInViewer(cloud_reg &cr);
};

void Context::updateInViewer(cloud_reg &cr)
{
    m_viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_OPACITY, cr.active ? 1.0 : 0.0,
        cr.name);
}

void Context::addCloud(cloud_reg &reg)
{
    // FSG_LOG_MSG("Adding cloud with key " << reg.key << ", name " <<
    // reg.name);
    FSG_LOG_MSG("Adding cloud " << reg);

    m_viewer->addPointCloud<pcl::PointXYZRGB>(reg.cloud, reg.name);

    m_clouds.push_front(reg);
    updateInViewer(reg);
    //    cloud_reg *newreg = &m_clouds.front();
}

void Context::handleKeyboardEvent(
    const pcl::visualization::KeyboardEvent &event)
{
    if (event.keyUp())
    {
        const std::string keySym = event.getKeySym();
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM("Handle key pressed '" << keySym
                                                                 << "'");

        if (keySym.compare("twosuperior") == 0)
        {
            FSG_LOG_MSG("Resetting camera.");
            m_viewer->setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0);
        }

        for (auto &cr : m_clouds)
        {
            // FSG_LOG_MSG("Checking key " << cr.key << ", name " << cr.name);
            if ((keySym.compare(cr.key) == 0))
            {
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
                           void *context_void)
{
    boost::shared_ptr<Context> context =
        *static_cast<boost::shared_ptr<Context> *>(context_void);

    context->handleKeyboardEvent(event);
}

Eigen::ComputationInfo minimizationResultToComputationInfo(
    Eigen::LevenbergMarquardtSpace::Status minimizationResult)
{
    switch (minimizationResult)
    {
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

void FloatTest()
{
    FSG_TRACE_THIS_FUNCTION();
    FNUM_TYPE ref = 1.0;
    FNUM_TYPE epsilon = ref;
    FNUM_TYPE ref_plus_epsilon = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    do
    {
        epsilon = epsilon / sg_2;
        // FSG_LOG_VAR(epsilon);
        ref_plus_epsilon = ref + epsilon;
        // FSG_LOG_VAR(ref_plus_epsilon);
    } while (ref_plus_epsilon != ref);
#pragma GCC diagnostic pop
    FSG_LOG_MSG("First epsilon that added to "
                << ref << " does not change a bit: " << epsilon);
}

static const FNUM_TYPE fit_control_epsilon = FNUM_LITERAL(0.01);

/**
   This method generates a number of varied parameter sets (trying
   first vanilla parameters then changing each dimensions in turn),
   and checks that each fits precisely the generated point cloud.
 */
void SuperEllipsoidTestEachDimensionForMisbehavior(
    fsg::SuperEllipsoidParameters &superellipsoidparameters_prototype)
{
    FSG_TRACE_THIS_FUNCTION();

    FSG_LOG_VAR(superellipsoidparameters_prototype);

    const pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud =
        superellipsoidparameters_prototype.toPointCloud(10);

    std::vector<int> indices(pointCloud->size());
    for (int i = 0; i < (int)pointCloud->size(); ++i)
    {
        indices[i] = i;
    }

    OptimizationFunctor functor(*pointCloud, indices);

    for (int dimension_shift = -1; dimension_shift < 11; dimension_shift++)
    {
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
            "SuperEllipsoidTestEachDimensionForMisbehavior dimension shift "
            << dimension_shift);
        fsg::SuperEllipsoidParameters superellipsoidparameters =
            superellipsoidparameters_prototype;

        if (dimension_shift >= 0)
        {
            superellipsoidparameters.coeff(dimension_shift) = 1.5;
        }

        VECTORX deviation(pointCloud->size());

        functor(superellipsoidparameters.coeff, deviation);

        // FSG_LOG_VAR(deviation);
        FSG_LOG_VAR(deviation.norm());

        if (deviation.norm() > fit_control_epsilon)
        {
            FSG_LOG_MSG("SuperEllipsoidTestEachDimensionForMisbehavior FAIL on "
                        "dimension "
                        << dimension_shift << ".");
        }
    }
}

void SuperEllipsoidTestComputeGradient(
    fsg::SuperEllipsoidParameters &superellipsoidparameters_prototype,
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud)
{

    FSG_LOG_VAR(superellipsoidparameters_prototype);

    std::vector<int> indices(pointCloud->size());
    for (int i = 0; i < (int)pointCloud->size(); ++i)
    {
        indices[i] = i;
    }

    OptimizationFunctor functor(*pointCloud, indices);

    VECTORX deviation(pointCloud->size());

    functor(superellipsoidparameters_prototype.coeff, deviation);

    FNUM_TYPE center_value = deviation.norm();

    FSG_LOG_VAR(center_value);

    const FNUM_TYPE epsilon = FNUM_LITERAL(0.05);

    for (int dimension_shift = 0; dimension_shift < 11; dimension_shift++)
    {
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
            "SuperEllipsoidTestComputeGradient dimension shift "
            << dimension_shift);
        fsg::SuperEllipsoidParameters superellipsoidparameters =
            superellipsoidparameters_prototype;

        superellipsoidparameters.coeff(dimension_shift) =
            superellipsoidparameters_prototype.coeff(dimension_shift) - epsilon;
        functor(superellipsoidparameters.coeff, deviation);
        FNUM_TYPE minus = deviation.norm();

        superellipsoidparameters.coeff(dimension_shift) =
            superellipsoidparameters_prototype.coeff(dimension_shift) + epsilon;
        functor(superellipsoidparameters.coeff, deviation);
        FNUM_TYPE plus = deviation.norm();

        FSG_LOG_MSG("dimension " << dimension_shift << " values " << minus << ""
                                 << center_value << "" << plus);
    }
}

bool pointCloudToFittingContextWithInitialEstimate_EigenLevenbergMarquardt(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz,
    fsg::SuperEllipsoidParameters &fittingContext)
{
    FSG_TRACE_THIS_FUNCTION();
    fsg::SuperEllipsoidParameters initialEstimate = fittingContext;

    FSG_LOG_MSG("Initial estimate : " << initialEstimate);

    std::vector<int> indices(cloud_xyz->size());
    for (int i = 0; i < (int)cloud_xyz->size(); ++i)
    {
        indices[i] = i;
    }

    OptimizationFunctor functor(*cloud_xyz, indices);
    Eigen::NumericalDiff<OptimizationFunctor> num_diff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<OptimizationFunctor>,
                              FNUM_TYPE>
        lm(num_diff);
    Eigen::LevenbergMarquardtSpace::Status minimizationResult;

    {
        FSG_TRACE_THIS_SCOPE_WITH_STATIC_STRING(
            "Eigen::LevenbergMarquardt::minimize()");
        VECTORX coeff_d = fittingContext.coeff;
        VECTORX coeff_f = coeff_d.cast<FNUM_TYPE>();

        minimizationResult = lm.minimize(coeff_f);
    }

    Eigen::ComputationInfo ci =
        minimizationResultToComputationInfo(minimizationResult);

    FSG_LOG_MSG("Minimization result: Eigen::ComputationInfo="
                << ci
                << " LevenbergMarquardtSpace=" << (int)minimizationResult);

    FSG_LOG_MSG("Initial estimation : " << initialEstimate);
    FSG_LOG_MSG("After minimization : " << fittingContext);

    if (ci != Eigen::ComputationInfo::Success)
    {
        FSG_LOG_MSG("Levenberg-Marquardt fitting failed, with code: " << ci);
        return false;
    }
    return true;
}

// bool pointCloudToFittingContextWithInitialEstimate_LibCmaes(
//     const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz,
//     fsg::SuperEllipsoidParameters &fittingContext)
// {
//     FSG_TRACE_THIS_FUNCTION();
//     fsg::SuperEllipsoidParameters initialEstimate = fittingContext;

//     FSG_LOG_MSG("Initial estimate : " << initialEstimate);

//     std::vector<int> indices(cloud_xyz->size());
//     for (int i = 0; i < (int)cloud_xyz->size(); ++i)
//     {
//         indices[i] = i;
//     }

//     std::vector<FNUM_TYPE> x0(fsg::SuperEllipsoidParameters::fieldCount, 0);

//     // copy to

//     OptimizationFunctor functor(*cloud_xyz, indices);

//     libcmaes::FitFunc cmaes_fit_func = [&functor, &cloud_xyz](const FNUM_TYPE
//     *x,
//                                                               const int N) {
//         fsg::SuperEllipsoidParameters cmaes_eval_params;

//         // FSG_LOG_VAR(N);

//         if (N != fsg::SuperEllipsoidParameters::fieldCount)
//         {
//             FSG_LOG_MSG("Mismatch field/vector count: "
//                         << fsg::SuperEllipsoidParameters::fieldCount << " vs.
//                         "
//                         << N);
//             exit(1);
//         }

//         for (int i = 0; i < N; i++)
//         {
//             cmaes_eval_params.coeff(i) = x[i];
//         }

//         // FSG_LOG_VAR(cmaes_eval_params);

//         VECTORX deviation(cloud_xyz->size());

//         functor(cmaes_eval_params.coeff, deviation);

//         FNUM_TYPE deviation_sum_of_squares = -deviation.norm();

//         // FSG_LOG_VAR(deviation_sum_of_squares);

//         return deviation_sum_of_squares;
//     };

//     VECTORX OptimizationStepSize(
//         fsg::SuperEllipsoidParameters::fieldCount);
//     OptimizationStepSize << FNUM_LITERAL(0.1), FNUM_LITERAL(0.1),
//     FNUM_LITERAL(0.1), FNUM_LITERAL(0.1), FNUM_LITERAL(0.1),
//     FNUM_LITERAL(0.1), FNUM_LITERAL(3.0), FNUM_LITERAL(1.5),
//     FNUM_LITERAL(1.5), sg_1, sg_1;

//     // FSG_LOG_VAR(OptimizationStepSize);

//     VECTORX LowerBounds(fsg::SuperEllipsoidParameters::fieldCount);
//     LowerBounds << FNUM_LITERAL(-3.0), FNUM_LITERAL(-3.0),
//     FNUM_LITERAL(-3.0), sg_0, sg_0, sg_0, -sg_pi, -sg_pi_2, -sg_pi_2,
//     FNUM_LITERAL(0.1), FNUM_LITERAL(0.1);

//     // FSG_LOG_VAR(LowerBounds);

//     VECTORX UpperBounds(fsg::SuperEllipsoidParameters::fieldCount);
//     UpperBounds << FNUM_LITERAL(3.0), FNUM_LITERAL(3.0), FNUM_LITERAL(3.0),
//     sg_1, sg_1, sg_1, sg_pi, sg_pi_2, sg_pi_2, FNUM_LITERAL(2.0),
//     FNUM_LITERAL(2.0);

//     // FSG_LOG_VAR(UpperBounds);

//     //
//     https://github.com/beniz/libcmaes/wiki/Defining-and-using-bounds-on-parameters
//     libcmaes::GenoPheno<libcmaes::pwqBoundStrategy> gp(
//         LowerBounds.data(), UpperBounds.data(),
//         fsg::SuperEllipsoidParameters::fieldCount); // genotype / phenotype
//                                                     // transform associated
//                                                     to
//                                                     // bounds.

//     libcmaes::CMAParameters<libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>>
//         cmaparams(fsg::SuperEllipsoidParameters::fieldCount,
//                   fittingContext.coeffData(), 3, 100, 0, gp);
//     // CMAParameters(const dVec &x0,
//     //               const dVec &sigma,
//     //               const int &lambda,
//     //               const dVec &lbounds,
//     //               const dVec &ubounds,
//     //               const uint64_t &seed);

//     {
//         static int n = 0;
//         auto libcmaes_graph_data_filename_stringstream = std::stringstream();
//         libcmaes_graph_data_filename_stringstream << "libcmaes_log_" << n
//                                                   << ".dat";
//         n++;
//         std::string libcmaes_graph_data_filename =
//             libcmaes_graph_data_filename_stringstream.str();
//         FSG_LOG_VAR(libcmaes_graph_data_filename);
//         cmaparams.set_fplot(libcmaes_graph_data_filename);
//     }

//     cmaparams.set_mt_feval(true); // activates the parallel evaluation
//     cmaparams.set_fixed_p(10, 1);
//     cmaparams.set_fixed_p(9, 1);
//     cmaparams.set_ftarget(0.001);
//     cmaparams.set_max_fevals(30000);
//     cmaparams.set_stopping_criteria(libcmaes::STAGNATION, false);
//     cmaparams.set_stopping_criteria(libcmaes::TOLX, false);
//     cmaparams.set_stopping_criteria(libcmaes::CONDITIONCOV, false);

//     cmaparams.set_algo(IPOP_CMAES);
//     cmaparams.set_ftolerance(1e-5);
//     cmaparams.set_xtolerance(1e-5);

//     FSG_LOG_VAR(cmaparams.get_x0min());
//     FSG_LOG_VAR(cmaparams.get_x0max());
//     FSG_LOG_VAR(cmaparams.get_max_iter());
//     FSG_LOG_VAR(cmaparams.get_max_fevals());
//     FSG_LOG_VAR(cmaparams.get_ftarget());
//     FSG_LOG_VAR(cmaparams.get_seed());
//     FSG_LOG_VAR(cmaparams.get_ftolerance());
//     FSG_LOG_VAR(cmaparams.get_xtolerance());
//     FSG_LOG_VAR(cmaparams.get_algo());
//     // FSG_LOG_VAR(cmaparams.get_gp());
//     FSG_LOG_VAR(cmaparams.get_fplot());
//     FSG_LOG_VAR(cmaparams.get_gradient());
//     FSG_LOG_VAR(cmaparams.get_edm());
//     FSG_LOG_VAR(cmaparams.get_mt_feval());
//     FSG_LOG_VAR(cmaparams.get_maximize());
//     FSG_LOG_VAR(cmaparams.get_uh());
//     FSG_LOG_VAR(cmaparams.get_tpa());
//     FSG_LOG_VAR(cmaparams.get_sigma_init());
//     FSG_LOG_VAR(cmaparams.get_restarts());
//     FSG_LOG_VAR(cmaparams.get_lazy_update());

//     libcmaes::CMASolutions cmasols =
//         libcmaes::cmaes<libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>>(
//             cmaes_fit_func, cmaparams);

//     FSG_LOG_MSG("best solution: " << cmasols);
//     FSG_LOG_MSG("optimization took " << cmasols.elapsed_time() / 1000.0
//                                      << " seconds\n");

//     int cma_status = cmasols.run_status();

//     FSG_LOG_VAR(cma_status); // the optimization status, failed if < 0
//     FSG_LOG_VAR(cmasols.status_msg());

//     //
//     https://github.com/beniz/libcmaes/wiki/Optimizing-a-function#user-content-solution-error-covariance-matrix-and-expected-distance-to-the-minimum-edm
//     libcmaes::Candidate bcand = cmasols.best_candidate();

//     FSG_LOG_BEGIN() << FSG_LOCATION() << FSG_INDENTATION()
//                     << "cmasols.print(..., 0, cmaparams.get_gp()): ";

//     cmasols.print(FSG_LOG_BEGIN(), false, cmaparams.get_gp());

//     FSG_LOG_BEGIN() << FSG_LOG_END();

//     FSG_LOG_BEGIN() << FSG_LOCATION() << FSG_INDENTATION()
//                     << "cmasols.print(..., 1, cmaparams.get_gp()): ";

//     cmasols.print(FSG_LOG_BEGIN(), false, cmaparams.get_gp());

//     FSG_LOG_BEGIN() << FSG_LOG_END();

//     // FNUM_TYPE fmin = bcand.get_fvalue(); // min objective function value
//     the
//     // optimizer converged to
//     VECTORX bestparameters = bcand.get_x_dvec(); // vector of objective
//                                                          // function
//                                                          parameters
//                                                          // at minimum.
//     // FNUM_TYPE edm = cmasols.edm(); // expected distance to the minimum.

//     //
//     https://github.com/beniz/libcmaes/wiki/Defining-and-using-bounds-on-parameters#user-content-retrieving-the-best-solution
//     // VECTORX bestparameters =
//     // gp.pheno(cmasols.get_best_seen_candidate().get_x_dvec());

//     fittingContext.coeff = bestparameters;

//     FSG_LOG_MSG("Initial estimation : " << initialEstimate);
//     FSG_LOG_MSG("After minimization : " << fittingContext);

//     {
//         /* Check  */
//         VECTORX deviation(cloud_xyz->size());
//         functor(fittingContext.coeff, deviation);
//         // FSG_LOG_VAR(deviation);
//         FSG_LOG_VAR(deviation.norm());
//     }

//     if (cma_status < 0)
//     {
//         FSG_LOG_MSG("CMAES fitting failed, with code: " << cma_status);
//         return false;
//     }
//     return true;
// }

void SuperEllipsoidComputeGradientAllDimensions(
    fsg::SuperEllipsoidParameters &superellipsoidparameters_center,
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud,
    fsg::SuperEllipsoidParameters *gradient)
{
    FSG_TRACE_THIS_FUNCTION();

    FSG_LOG_VAR(superellipsoidparameters_center);

    std::vector<int> indices(pointCloud->size());
    for (int i = 0; i < (int)pointCloud->size(); ++i)
    {
        indices[i] = i;
    }

    OptimizationFunctor functor(*pointCloud, indices);

    VECTORX deviation(pointCloud->size());

    VECTORX values(5);
    const FNUM_TYPE epsilon = FNUM_LITERAL(0.001);

    for (int dimension_shift = 0; dimension_shift < 11; dimension_shift++)
    {
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
            "SuperEllipsoidComputeGradientAllDimensions dimension shift "
            << dimension_shift);

        for (int step = -1; step <= 1; step += 1)
        {
            FNUM_TYPE stepEpsilon = (FNUM_TYPE)step * epsilon;

            fsg::SuperEllipsoidParameters superellipsoidparameters =
                superellipsoidparameters_center;

            superellipsoidparameters.coeff(dimension_shift) += stepEpsilon;

            FSG_LOG_VAR(stepEpsilon);
            FSG_LOG_VAR(superellipsoidparameters);

            functor(superellipsoidparameters.coeff, deviation);

            // FSG_LOG_VAR(deviation);
            FNUM_TYPE value = deviation.norm();
            FSG_LOG_VAR(value);

            values[step + 1] = value;
        }

        gradient->coeff(dimension_shift) = values[2] - values[0];
    }
    FSG_LOG_VAR(*gradient);
}

bool pointCloudToFittingContextWithInitialEstimate_both(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz,
    fsg::SuperEllipsoidParameters &fittingContext)
{

    // fsg::SuperEllipsoidParameters gradient_wrt_pointcloud;
    // SuperEllipsoidComputeGradientAllDimensions(fittingContext, cloud_xyz,
    //                                            &gradient_wrt_pointcloud);

    // fsg::SuperEllipsoidParameters fittingContext_eigenlevenbergmarquardt(
    //     fittingContext);
    // bool success_eigenlevenbergmarquardt =
    //     pointCloudToFittingContextWithInitialEstimate_EigenLevenbergMarquardt(
    //         cloud_xyz, fittingContext_eigenlevenbergmarquardt);
    // fsg::SuperEllipsoidParameters fittingContext_libcmaes(fittingContext);
    // bool success_libcmaes =
    //     pointCloudToFittingContextWithInitialEstimate_LibCmaes(
    //         cloud_xyz, fittingContext_libcmaes);
    // FSG_LOG_MSG("pointCloudToFittingContextWithInitialEstimate_both results:
    // "
    //             "EigenLevenbergMarquardt="
    //             << (success_eigenlevenbergmarquardt ? "success" : "failure")
    //             << ", libcmaes result="
    //             << (success_libcmaes ? "success" : "failure"));

    // if (1)
    // {
    //     boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
    //         new pcl::visualization::PCLVisualizer("viewer"));
    //     viewer->setBackgroundColor(0, 0, 0);
    //     viewer->setCameraPosition(-3, 0, 0, 1, 0, 0, 0, 1, 0);
    //     viewer->addCoordinateSystem(1.0);

    //     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
    //         ground_truth_color_handler(cloud_xyz, 0, 255, 0);
    //     viewer->addPointCloud<pcl::PointXYZ>(
    //         cloud_xyz, ground_truth_color_handler, "ground_truth");
    //     viewer->setPointCloudRenderingProperties(
    //         pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
    //         "ground_truth");

    //     auto pc_seplce = fittingContext_libcmaes.toPointCloud(10);

    //     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
    //         sep_libcmaes_color_handler(pc_seplce, 255, 0, 255);
    //     viewer->addPointCloud<pcl::PointXYZ>(
    //         pc_seplce, sep_libcmaes_color_handler, "sep_libcmaes");
    //     viewer->setPointCloudRenderingProperties(
    //         pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
    //         "sep_libcmaes");

    //     auto pc_elm =
    //     fittingContext_eigenlevenbergmarquardt.toPointCloud(10);

    //     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
    //         sep_levenbergmarquardt_color_handler(pc_elm, 255, 0, 0);
    //     viewer->addPointCloud<pcl::PointXYZ>(
    //         pc_elm, sep_levenbergmarquardt_color_handler,
    //         "sep_levenbergmarquardt");
    //     viewer->setPointCloudRenderingProperties(
    //         pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
    //         "sep_levenbergmarquardt");

    //     for (int i = 0; i < 3; i++)
    //     {
    //         viewer->spinOnce(100);
    //         boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    //     }

    //     {
    //         static int n = 0;
    //         auto libcmaes_png_filename_stringstream = std::stringstream();
    //         libcmaes_png_filename_stringstream << "libcmaes_png_" << n
    //                                            << ".png";
    //         n++;
    //         std::string libcmaes_png_filename =
    //             libcmaes_png_filename_stringstream.str();
    //         FSG_LOG_VAR(libcmaes_png_filename);
    //         viewer->saveScreenshot(libcmaes_png_filename);
    //     }
    //     viewer->close();
    // }

    // if (success_eigenlevenbergmarquardt)
    // {
    //     fittingContext = fittingContext_eigenlevenbergmarquardt;
    //     FSG_LOG_MSG("pointCloudToFittingContextWithInitialEstimate_both: "
    //                 "returning EigenLevenbergMarquardt result");
    //     return true;
    // }

    // if (success_libcmaes)
    // {
    //     fittingContext = fittingContext_libcmaes;
    //     FSG_LOG_MSG("pointCloudToFittingContextWithInitialEstimate_both: "
    //                 "returning libcmaes result");
    //     return true;
    // }

    // FSG_LOG_MSG("pointCloudToFittingContextWithInitialEstimate_both: both "
    //             "failed.  Returning initial estimation.");
    return false;
}

bool pointCloudToFittingContextWithInitialEstimate(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz,
    fsg::SuperEllipsoidParameters &fittingContext)
{

    {
        std::vector<int> indices(cloud_xyz->size());
        for (int i = 0; i < (int)cloud_xyz->size(); ++i)
        {
            indices[i] = i;
        }

        OptimizationFunctor functor(*cloud_xyz, indices);

        VECTORX deviation(cloud_xyz->size());

        functor(fittingContext.coeff, deviation);

        // FSG_LOG_VAR(deviation);
        FSG_LOG_VAR(deviation.norm());
        FSG_LOG_MSG("pointCloudToFittingContextWithInitialEstimate: initial "
                    "estimate has deviation.norm() = "
                    << deviation.norm());
    }

    return pointCloudToFittingContextWithInitialEstimate_EigenLevenbergMarquardt(
        cloud_xyz, fittingContext);
}

/**
    Given a parameter set and a point cloud which exactly matches,
    verify that disturbing in any dimension yields a sane gradient
    that converges back to the original parameter set.
*/
void SuperEllipsoidTestEachDimensionForGradientSanity(
    fsg::SuperEllipsoidParameters &superellipsoidparameters_center,
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud)
{
    FSG_TRACE_THIS_FUNCTION();

    FSG_LOG_VAR(superellipsoidparameters_center);

    std::vector<int> indices(pointCloud->size());
    for (int i = 0; i < (int)pointCloud->size(); ++i)
    {
        indices[i] = i;
    }

    OptimizationFunctor functor(*pointCloud, indices);

    VECTORX deviation(pointCloud->size());

    VECTORX values(5);
    const FNUM_TYPE epsilon = FNUM_LITERAL(0.001);

    functor(superellipsoidparameters_center.coeff, deviation);
    FNUM_TYPE centervalue = deviation.norm();
    FSG_LOG_VAR(centervalue);
    values[2] = centervalue;

    for (int dimension_shift = 0; dimension_shift < 11; dimension_shift++)
    {
        FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
            "SuperEllipsoidTestEachDimensionForGradientSanity dimension shift "
            << dimension_shift);

        /* First check that gradient itself is good. */
        for (int step = -2; step <= 2; step++)
        {
            /*if (step == 0)
            {
                continue;
            }*/
            FNUM_TYPE stepEpsilon = (FNUM_TYPE)step * epsilon;

            fsg::SuperEllipsoidParameters superellipsoidparameters =
                superellipsoidparameters_center;

            superellipsoidparameters.coeff(dimension_shift) += stepEpsilon;

            // FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
            //     "dimension shift " << dimension_shift << " stepEpsilon "
            //                        << stepEpsilon);
            FSG_LOG_VAR(stepEpsilon);
            FSG_LOG_VAR(superellipsoidparameters);

            functor(superellipsoidparameters.coeff, deviation);

            // FSG_LOG_VAR(deviation);
            FNUM_TYPE value = deviation.norm();
            FSG_LOG_VAR(value);

            values[step + 2] = value;
        }

        FSG_LOG_VAR(values);
        FSG_LOG_MSG("values=[" << values[0] << ", " << values[1] << ",  "
                               << values[2] << "  , " << values[3] << ", "
                               << values[4] << "]");

        {
            int failures = 0;
            if (values[0] < values[1])
            {
                FSG_LOG_MSG("FAIL: bad gradient lower side on dimension "
                            << dimension_shift << " values " << values
                            << " params " << superellipsoidparameters_center);
                failures++;
            }

            if (values[3] > values[4])
            {
                FSG_LOG_MSG("FAIL: bad gradient higher side on dimension "
                            << dimension_shift << " values " << values
                            << " params " << superellipsoidparameters_center);
                failures++;
            }

            if (values[2] > values[1])
            {
                FSG_LOG_MSG(
                    "FAIL: bad gradient center higher than left on dimension "
                    << dimension_shift << " values " << values << " params "
                    << superellipsoidparameters_center);
                failures++;
            }

            if (values[2] > values[3])
            {
                FSG_LOG_MSG(
                    "FAIL: bad gradient center higher than right on dimension "
                    << dimension_shift << " values " << values << " params "
                    << superellipsoidparameters_center);
                failures++;
            }

            if (failures == 0)
            {
                FSG_LOG_MSG("Good gradient on dimension "
                            << dimension_shift << " params "
                            << superellipsoidparameters_center << " values "
                            << values);
            }
        }

        {
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

            if (!success)
            {
                FSG_LOG_MSG(
                    "Fit after gradient epsilon failed, thus gradient test "
                    "FAIL, on dimension "
                    << dimension_shift);
                continue;
            }

            FSG_LOG_MSG("Fit after gradient epsilon converges, on dimension "
                        << dimension_shift);
            FSG_LOG_VAR(superellipsoidparameters_fit);
            FSG_LOG_VAR(superellipsoidparameters_center);

            VECTORX deviation = superellipsoidparameters_fit.coeff -
                                superellipsoidparameters_center.coeff;

            FSG_LOG_VAR(deviation.norm());

            if (deviation.norm() > fit_control_epsilon)
            {
                FSG_LOG_MSG("Test FAIL fitting parameter at epsilon deviation"
                            "on dimension "
                            << dimension_shift);
            }
        }
    }
}

fsg::SuperEllipsoidParameters pointCloudComputeFitComputeInitialEstimate(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz,
    pcl::visualization::PCLVisualizer *viewer, const std::string &obj_index_i_s)
{
    FSG_TRACE_THIS_FUNCTION();
    fsg::SuperEllipsoidParameters initialEstimate;
    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(cloud_xyz);
    // Minimize eccentricity computation.
    feature_extractor.setAngleStep(360);
    feature_extractor.compute();

    Eigen::Vector3f
        mass_center; // Type imposed by pcl::MomentOfInertiaEstimation
    feature_extractor.getMassCenter(
        mass_center); // FIXME should check return value

    FSG_LOG_VAR(mass_center);

    Eigen::Vector3f major_vector, middle_vector,
        minor_vector; // Type imposed by pcl::MomentOfInertiaEstimation

    feature_extractor.getEigenVectors(major_vector, middle_vector,
                                      minor_vector);
    FSG_LOG_VAR(major_vector);
    FSG_LOG_VAR(middle_vector);
    FSG_LOG_VAR(minor_vector);

    pcl::PointXYZ min_point_OBB;
    pcl::PointXYZ max_point_OBB;
    pcl::PointXYZ position_OBB;
    Eigen::Matrix3f
        rotational_matrix_OBB; // Type imposed by pcl::MomentOfInertiaEstimation
    feature_extractor.getOBB(
        min_point_OBB, max_point_OBB, position_OBB,
        rotational_matrix_OBB); // FIXME should check return value

    FSG_LOG_VAR(position_OBB);
    FSG_LOG_VAR(min_point_OBB);
    FSG_LOG_VAR(max_point_OBB);

    FSG_LOG_VAR(rotational_matrix_OBB);

    // FIXME clarify/generalize major/z.
    major_vector *= (max_point_OBB.z - min_point_OBB.z) / 2.0f;
    middle_vector *= (max_point_OBB.y - min_point_OBB.y) / 2.0f;
    minor_vector *= (max_point_OBB.x - min_point_OBB.x) / 2.0f;
    FSG_LOG_VAR(major_vector);
    FSG_LOG_VAR(middle_vector);
    FSG_LOG_VAR(minor_vector);

    // From
    // http://pointclouds.org/documentation/tutorials/moment_of_inertia.php

    Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
    Eigen::Quaternionf quat(rotational_matrix_OBB);

    pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));

    // FIXME clarify/generalize major/z.
    pcl::PointXYZ maj_axis(2.0f * major_vector(0) + mass_center(0),
                           2.0f * major_vector(1) + mass_center(1),
                           2.0f * major_vector(2) + mass_center(2));
    pcl::PointXYZ mid_axis(2.0f * middle_vector(0) + mass_center(0),
                           2.0f * middle_vector(1) + mass_center(1),
                           2.0f * middle_vector(2) + mass_center(2));
    pcl::PointXYZ min_axis(2.0f * minor_vector(0) + mass_center(0),
                           2.0f * minor_vector(1) + mass_center(1),
                           2.0f * minor_vector(2) + mass_center(2));

    if (viewer != NULL)
    {
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

    initialEstimate.set_cen_x(mass_center(0));
    initialEstimate.set_cen_y(mass_center(1));
    initialEstimate.set_cen_z(mass_center(2));

    initialEstimate.set_rad_a(major_vector.norm());

    initialEstimate.set_rad_b(middle_vector.norm());

    initialEstimate.set_rad_c(minor_vector.norm());

    FNUM_TYPE yaw, pitch, roll;

    // https://stackoverflow.com/questions/24764031/cast-eigenmatrixxd-to-eigenmatrixxf
    MATRIX3 rotational_matrix_OBB_FNUM =
        rotational_matrix_OBB.cast<FNUM_TYPE>();
    matrix_to_angles(rotational_matrix_OBB_FNUM, yaw, pitch, roll);

    FSG_LOG_MSG("yaw=" << yaw << ", pitch=" << pitch << ", roll=" << roll);

    initialEstimate.set_rot_yaw(yaw);
    initialEstimate.set_rot_pitch(pitch);
    initialEstimate.set_rot_roll(roll);

    initialEstimate.set_exp_1(sg_1);
    initialEstimate.set_exp_2(sg_1);

    return initialEstimate;
}

void SuperEllipsoidGraphFitnessLandscapeSliceBetweenPositions(
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud,
    fsg::SuperEllipsoidParameters sep_initialEstimate,
    fsg::SuperEllipsoidParameters sep_groundtruth)
{
    FSG_TRACE_THIS_FUNCTION();

    static ofstream *slicelog = NULL;
    if (slicelog == NULL)
    {
        slicelog = new ofstream(
            "SuperEllipsoidGraphFitnessLandscapeSliceBetweenPositions.log");
        FSG_LOG_VAR(slicelog);
    }

    (*slicelog) << sep_initialEstimate << sep_groundtruth << " ";

    std::vector<int> indices(pointCloud->size());
    for (int i = 0; i < (int)pointCloud->size(); ++i)
    {
        indices[i] = i;
    }

    FSG_LOG_VAR(sep_initialEstimate);
    FSG_LOG_VAR(sep_groundtruth);
    fsg::SuperEllipsoidParameters sep_current;
    const int steps = 1000;
    for (int step = 0; step <= steps; step++)
    {
        FNUM_TYPE d = (FNUM_TYPE)step / (FNUM_TYPE)steps;
        FSG_LOG_VAR(d);
        sep_current = (sg_1 - d) * sep_initialEstimate + d * sep_groundtruth;

        FSG_LOG_VAR(sep_current);

        OptimizationFunctor functor(*pointCloud, indices);

        VECTORX deviation(pointCloud->size());

        functor(sep_current.coeff, deviation);

        FNUM_TYPE dn = deviation.norm();
        // FSG_LOG_VAR(deviation);
        FSG_LOG_VAR(dn);

        (*slicelog) << "d= " << d << " dn= " << dn << " ";
    }
    (*slicelog) << "\n";
    slicelog->flush();
}

bool SuperEllipsoidFitARandomSQ(boost::random::minstd_rand &_gen)
{
    FSG_TRACE_THIS_FUNCTION();
    boost::random::uniform_real_distribution<> random_number_m5p5(-1, 1);
    boost::random::uniform_real_distribution<> random_number_cent_one(0.01, 1);
    boost::random::uniform_real_distribution<> random_number_mpippi(-sg_pi,
                                                                    sg_pi);
    // ost::random::uniform_real_distribution<> random_number_cent_two(1, 1);

    fsg::SuperEllipsoidParameters sep_groundtruth;
    sep_groundtruth.set_cen_x((FNUM_TYPE)random_number_m5p5(_gen));
    sep_groundtruth.set_cen_y((FNUM_TYPE)random_number_m5p5(_gen));
    sep_groundtruth.set_cen_z((FNUM_TYPE)random_number_m5p5(_gen));
    sep_groundtruth.set_rad_a((FNUM_TYPE)random_number_cent_one(_gen));
    sep_groundtruth.set_rad_b((FNUM_TYPE)random_number_cent_one(_gen));
    sep_groundtruth.set_rad_c((FNUM_TYPE)random_number_cent_one(_gen));
    sep_groundtruth.set_rot_yaw((FNUM_TYPE)random_number_mpippi(_gen));
    sep_groundtruth.set_rot_pitch((FNUM_TYPE)random_number_mpippi(_gen));
    sep_groundtruth.set_rot_roll((FNUM_TYPE)random_number_mpippi(_gen));
    sep_groundtruth.set_exp_1(
        sg_1); // (FNUM_TYPE)random_number_cent_two(_gen));
    sep_groundtruth.set_exp_2(
        sg_1); // (FNUM_TYPE)random_number_cent_two(_gen));

    FSG_LOG_VAR(sep_groundtruth);

    SuperEllipsoidTestEachDimensionForMisbehavior(sep_groundtruth);

    FSG_LOG_VAR(sep_groundtruth);

    const pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud =
        sep_groundtruth.toPointCloud(4);

    SuperEllipsoidTestEachDimensionForGradientSanity(sep_groundtruth,
                                                     pointCloud);

    FSG_LOG_MSG("Now testing actual fit.");
    FSG_LOG_VAR(sep_groundtruth);

    fsg::SuperEllipsoidParameters initialEstimate =
        pointCloudComputeFitComputeInitialEstimate(pointCloud, nullptr, "");

    SuperEllipsoidGraphFitnessLandscapeSliceBetweenPositions(
        pointCloud, initialEstimate, sep_groundtruth);

    {
        fsg::SuperEllipsoidParameters sep_fit = initialEstimate;

        bool success =
            pointCloudToFittingContextWithInitialEstimate(pointCloud, sep_fit);

        if (!success)
        {
            FSG_LOG_MSG("Fit failed, thus test FAIL, on parameter: "
                        << sep_groundtruth);
            return false;
        }

        FSG_LOG_VAR(sep_groundtruth);
        FSG_LOG_VAR(sep_fit);

        VECTORX deviation = sep_fit.coeff - sep_groundtruth.coeff;

        FSG_LOG_VAR(deviation.norm());

        if (deviation.norm() > fit_control_epsilon)
        {
            FSG_LOG_MSG("Test FAIL fitting parameter: " << sep_groundtruth);
            return false;
        }
        FSG_LOG_MSG("Test success fitting parameter: " << sep_groundtruth);
    }
    return true;
}

void SuperEllipsoidTestSlicebetweenPoints(FNUM_TYPE xmin, FNUM_TYPE xmax)
{
    fsg::SuperEllipsoidParameters unit_sphere =
        fsg::SuperEllipsoidParameters::Zero();
    unit_sphere.set_rad_a(1.0);
    unit_sphere.set_rad_b(1.0);
    unit_sphere.set_rad_c(1.0);
    unit_sphere.set_exp_1(1.0);
    unit_sphere.set_exp_2(1.0);
    const pcl::PointCloud<pcl::PointXYZ>::Ptr unit_sphere_point_cloud =
        unit_sphere.toPointCloud(10);

    fsg::pointCloudLogSomeVariables(unit_sphere_point_cloud);

    fsg::SuperEllipsoidParameters side_sphere_m = unit_sphere;
    side_sphere_m.set_cen_x(xmin);

    fsg::pointCloudLogSomeVariables(side_sphere_m.toPointCloud(10));

    fsg::SuperEllipsoidParameters side_sphere_p = unit_sphere;
    side_sphere_p.set_cen_x(xmax);

    fsg::pointCloudLogSomeVariables(side_sphere_p.toPointCloud(10));

    std::vector<int> indices(unit_sphere_point_cloud->size());
    for (int i = 0; i < (int)unit_sphere_point_cloud->size(); ++i)
    {
        indices[i] = i;
    }

    OptimizationFunctor functor(*unit_sphere_point_cloud, indices);

    VECTORX deviation(unit_sphere_point_cloud->size());

    functor(side_sphere_m.coeff, deviation);
    FSG_LOG_VAR(deviation.norm());
    functor(side_sphere_p.coeff, deviation);
    FSG_LOG_VAR(deviation.norm());

    SuperEllipsoidGraphFitnessLandscapeSliceBetweenPositions(
        unit_sphere_point_cloud, side_sphere_m, side_sphere_p);
}

void SuperEllipsoidTest()
{
    // SuperEllipsoidTestSlicebetweenPoints(-10, 10);
    // SuperEllipsoidTestSlicebetweenPoints(-4, 4);
    // SuperEllipsoidTestSlicebetweenPoints(-1, 1);

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
        for (int i = 0; i < 100; i++)
        {
            FSG_TRACE_THIS_SCOPE_WITH_SSTREAM("Fitting random " << i);
            SuperEllipsoidFitARandomSQ(_gen);
        }
    }
}

int main(int argc, char **argv)
{
    FSG_LOG_INIT__CALL_FROM_CPP_MAIN();
    FSG_LOG_MSG("");
    FSG_LOG_MSG("");
    FSG_LOG_MSG("===========================================================");
    FSG_LOG_VAR(GIT_VERSION);
    FSG_LOG_VAR(GIT_LOG);
    FSG_LOG_MSG("===========================================================");
    FSG_LOG_MSG("");
    FSG_LOG_MSG("");
    FSG_TRACE_THIS_FUNCTION();

#define e4b5877050777b69_STR(x) #x
#define e4b5877050777b69_xSTR(x) e4b5877050777b69_STR(x)
    FSG_LOG_MSG("Compiled with float type: " e4b5877050777b69_xSTR(FNUM_TYPE));

    if (argc == 2)
    {
        std::string test = "test";
        if (test.compare(argv[1]) == 0)
        {
            FloatTest();
            SuperEllipsoidTest();
            return 0;
        }
    }

    if (argc != 4)
    {

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
            if (!ifs)
            {
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
                     it_p != input_cloud_soi->end(); it_p++)
                {
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
                 it_sv++)
            {
                // int current_sv_label = it_sv->first;
                pcl::Supervoxel<ip::PointT>::Ptr current_sv = it_sv->second;
                double c = weights_for_this_modality[it_sv->first][lbl];

                if (c < 0.5)
                {
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
                uint8_t r = uint8_t(dist(_gen) * 56 * (c + 0.5));
                uint8_t g = uint8_t(dist(_gen) * 56 * (c + 0.5));
                uint8_t b = uint8_t(dist(_gen) * 56 * (c + 0.5));

                pcl::PointXYZRGB pt;
                for (auto v : *(current_sv->voxels_))
                {
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
                 obj_hypotheses | boost::adaptors::indexed(0))
            {

                std::string obj_index_i_s = std::to_string(obj_hyp.index());
                FSG_TRACE_THIS_SCOPE_WITH_SSTREAM(
                    "Considering obj hypothesis id=" << obj_index_i_s);

                std::set<uint32_t> *p_obj_hyp = &(obj_hyp.value());

                uint8_t r = uint8_t(dist(_gen) * 85);
                uint8_t g = uint8_t(dist(_gen) * 85);
                uint8_t b = uint8_t(dist(_gen) * 85);

                FSG_LOG_MSG("Assigned color = " << r << "," << g << "," << b);

                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(
                    new pcl::PointCloud<pcl::PointXYZ>);

                if (p_obj_hyp->size() <= 1)
                {
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
                         it_sv != supervoxels.end(); it_sv++)
                    {
                        int current_sv_label = it_sv->first;
                        pcl::Supervoxel<ip::PointT>::Ptr current_sv =
                            it_sv->second;

                        if (p_obj_hyp->find(current_sv_label) ==
                            p_obj_hyp->end())
                        {
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
                        for (auto v : *(current_sv->voxels_))
                        {
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

                if (cloud_xyz->size() < 20)
                {
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

                    for (auto v : *cloud_xyz)
                    {
                        pt.x = v.x;
                        pt.y = v.y;
                        pt.z = v.z;

                        pt.r = r;
                        pt.g = g;
                        pt.b = b;
                        object_hyps_cloud_ptr->push_back(pt);
                    }
                }

                fsg::SuperEllipsoidParameters initialEstimate =
                    pointCloudComputeFitComputeInitialEstimate(
                        cloud_xyz, &(*viewer), obj_index_i_s);

                fsg::SuperEllipsoidParameters fittingContext = initialEstimate;

                bool success = pointCloudToFittingContextWithInitialEstimate(
                    cloud_xyz, fittingContext);

                if (success)
                {

                    pcl::PointCloud<pcl::PointXYZ>::Ptr proj_points =
                        fittingContext.toPointCloud(100);

                    {
                        pcl::PointXYZRGB pt;

                        r = (uint8_t)(127 + (r >> 1));
                        g = (uint8_t)(127 + (g >> 1));
                        b = (uint8_t)(127 + (b >> 1));
                        for (auto v : *proj_points)
                        {
                            pt.x = v.x;
                            pt.y = v.y;
                            pt.z = v.z;

                            pt.r = r;
                            pt.g = g;
                            pt.b = b;
                            superellipsoids_cloud_ptr->push_back(pt);
                        }
                    }
                }
                else
                {
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

        for (auto &cr : clouds)
        {
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

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    viewer->close();

    return 0;
}
