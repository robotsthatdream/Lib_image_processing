#define _USE_MATH_DEFINES
#include "test_rotation.hpp"
#include "gtest/gtest.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace fsg {
namespace matrixrotationangles {

/* See documentation for the following topics in test_rotation.hpp:

   - Definition of rotational matrix

   - Definition of coordinate system: axis convention

   - Concrete illustration of axis convention

   - Definition of angle convention

 */

void matrix_to_angles(const Eigen::Matrix3f &m, float &yaw, float &pitch,
                      float &roll) {
    /* Ok, so how do we compute our angles?

       Yaw is the one we can compute first.

       It's the angle from X to the projection of its image on the XY plane.

       The image of X is in the first column of the matrix.

       Projection of its image on the XY plane, so, we'll use only
       m(0,0) and m(1,0).

       Note: in case pitch is +-PI, the image of X has both X and Y
       components equal to 0.  From "man atan2":

       > If y is +0 (-0) and x is +0, +0 (-0) is returned.

       0 is an acceptable value in this case.

    */
    yaw = atan2f(m(1, 0), m(0, 0));

    const Eigen::Matrix3f m_without_yaw =
        Eigen::AngleAxisf(-yaw, Eigen::Vector3f::UnitZ()) * m;

    // std::cerr << "m_without_yaw" << std::endl << m_without_yaw << std::endl;

    /* Now we want pitch.  We know the matrix no longer has yaw.

       It's the angle from X to the projection of its image on the XZ plane.

       In other words, it's the atan2 (component of image of X on Z, component
       of image of X on X).
    */

    pitch = atan2(m_without_yaw(2, 0), m_without_yaw(0, 0));

    const Eigen::Matrix3f m_without_yaw_nor_pitch =
        Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) * m_without_yaw;

    // std::cerr << "m_without_yaw_nor_pitch" << std::endl <<
    // m_without_yaw_nor_pitch << std::endl;

    /* Ok, we're nearly there.  We know the matrix is only roll now.

       It's the angle from Y to the projection of its image on the YZ plane.

       In other words, it's the atan2 (component of image of Y on Z, component
       of image of Y on Y).

     */

    roll = atan2(m_without_yaw_nor_pitch(2, 1), m_without_yaw_nor_pitch(1, 1));
}

void angles_to_matrix(const float &yaw, const float &pitch, const float &roll,
                      Eigen::Matrix3f &m) {
    // std::cerr << __PRETTY_FUNCTION__ << " yaw=" << yaw << " pitch=" << pitch
    //           << " roll=" << roll << std::endl;
    m = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(-pitch, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
    // std::cerr << m << std::endl << "is unitary: " << m.isUnitary() <<
    // std::endl;
}
}
}

// =====================================================================
// Test below
// =====================================================================

#ifdef FSG_COMPILING_GTEST_EXECUTABLE

using namespace fsg::matrixrotationangles;

void expect_identical_3x3_matrices(const Eigen::Matrix3f &m1,
                                   const Eigen::Matrix3f &m2,
                                   const float epsilon = 1e-5) {
    for (int row = 0; row <= 2; row++) {
        for (int col = 0; col <= 2; col++) {
            EXPECT_NEAR(m1(row, col), m2(row, col), epsilon) << "row=" << row
                                                             << ", col=" << col;
        }
    }
}

bool check_if_unitary(const Eigen::Matrix3f &m) {
    for (int i = 0; i < 3; ++i) {
        std::cerr << "norm(col(" << i << ")) = " << m.col(i).squaredNorm()
                  << std::endl;
        for (int j = 0; j < i; ++j) {
            std::cerr << "(col(" << i << ")).(col(" << j
                      << ")) = " << m.col(i).dot(m.col(j)) << std::endl;
        }
    }
    return true;
}

class TwoWayTest : public ::testing::Test {
  protected:
    Eigen::Matrix3f model_m;
    float model_yaw, model_pitch, model_roll;

    void CheckTwoWays() {
        // std::cerr << model_m << std::endl
        //           << " model_yaw=" << model_yaw
        //           << " model_pitch=" << model_pitch
        //           << " model_roll=" << model_roll << std::endl;

        if (!model_m.isUnitary()) {
            check_if_unitary(model_m);
        }

        ASSERT_TRUE(model_m.isUnitary());

        float yaw_computed, pitch_computed, roll_computed;
        matrix_to_angles(model_m, yaw_computed, pitch_computed, roll_computed);

        EXPECT_NEAR(model_yaw, yaw_computed, 1e-5);
        EXPECT_NEAR(model_pitch, pitch_computed, 1e-5);
        EXPECT_NEAR(model_roll, roll_computed, 1e-5);

        Eigen::Matrix3f m_computed;
        angles_to_matrix(model_yaw, model_pitch, model_roll, m_computed);
        expect_identical_3x3_matrices(model_m, m_computed);
    }
};

TEST(RotMatTest, EigenTest_CoeffAccessIsMRowColumn) {
    Eigen::Matrix3f m;

    m.row(0) << 1, 2, 3;
    m.row(1) << 4, 5, 6;
    m.row(2) << 7, 8, 9;

    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(1, 0), 4);
    EXPECT_EQ(m(2, 0), 7);
    EXPECT_EQ(m(2, 1), 8);
    EXPECT_EQ(m(2, 2), 9);
}

TEST_F(TwoWayTest, Identity) {
    model_m.row(0) << 1, 0, 0;
    model_m.row(1) << 0, 1, 0;
    model_m.row(2) << 0, 0, 1;

    model_yaw = model_pitch = model_roll = 0;

    CheckTwoWays();
}

TEST_F(TwoWayTest,
       YawTest_RotateBookCounterClockwiseQuarterTurnMustYieldYawPi2) {
    // This matrix sends X to Y.
    // This matrix sends Y to -X.
    // This matrix sends Z to Z.
    model_m.row(0) << 0, -1, 0;
    model_m.row(1) << 1, 0, 0;
    model_m.row(2) << 0, 0, 1;

    model_yaw = M_PI_2;
    model_pitch = model_roll = 0;

    CheckTwoWays();
}

#define M_SQRT2_2 ((M_SQRT2) / 2.0)

TEST_F(TwoWayTest, YawTest_RotateBookCounterClockwiseEigthTurnMustYieldYawPi4) {
    // This matrix sends X to Y.
    // This matrix sends Y to -X.
    // This matrix sends Z to Z.
    model_m.row(0) << M_SQRT2_2, -M_SQRT2_2, 0;
    model_m.row(1) << M_SQRT2_2, M_SQRT2_2, 0;
    model_m.row(2) << 0, 0, 1;

    model_yaw = M_PI_4;
    model_pitch = model_roll = 0;

    CheckTwoWays();
}

TEST_F(TwoWayTest, PitchTest_LiftBookPagetopQuarterTurnMustYieldPitchPi2) {
    // This matrix sends X to Z.
    // This matrix sends Y to Y.
    // This matrix sends Z to -X.
    model_m.row(0) << 0, 0, -1;
    model_m.row(1) << 0, 1, 0;
    model_m.row(2) << 1, 0, 0;

    model_yaw = model_roll = 0;
    model_pitch = M_PI_2;

    CheckTwoWays();
}

TEST_F(TwoWayTest, PitchTest_LiftBookPagetopEighthTurnMustYieldPitchPi4) {
    // This matrix sends X to Z.
    // This matrix sends Y to Y.
    // This matrix sends Z to -X.
    model_m.row(0) << M_SQRT2_2, 0, -M_SQRT2_2;
    model_m.row(1) << 0, 1, 0;
    model_m.row(2) << M_SQRT2_2, 0, M_SQRT2_2;

    model_yaw = model_roll = 0;
    model_pitch = M_PI_4;

    CheckTwoWays();
}

TEST_F(TwoWayTest, RollTest_OpenBookCoverQuarterTurnMustYieldRollMinusPi2) {
    // This matrix sends X to X.
    // This matrix sends Y to -Z.
    // This matrix sends Z to Y.
    model_m.row(0) << 1, 0, 0;
    model_m.row(1) << 0, 0, 1;
    model_m.row(2) << 0, -1, 0;

    model_yaw = model_pitch = 0;
    model_roll = -M_PI_2;

    CheckTwoWays();
}

TEST_F(TwoWayTest, RollTest_OpenBookCoverEighthTurnMustYieldRollMinusPi4) {
    // This matrix sends X to X.
    // This matrix sends Y to -Z.
    // This matrix sends Z to Y.
    model_m.row(0) << 1, 0, 0;
    model_m.row(1) << 0, M_SQRT2_2, M_SQRT2_2;
    model_m.row(2) << 0, -M_SQRT2_2, M_SQRT2_2;

    model_yaw = model_pitch = 0;
    model_roll = -M_PI_4;

    CheckTwoWays();
}

TEST_F(TwoWayTest, YawAndHalfPitchTest) {
    // This matrix sends X to (Y+Z)/SQRT2.
    // This matrix sends Y to -X.
    // This matrix sends Z to (-Y+Z)/SQRT2.
    model_m.row(0) << 0, -1, 0;
    model_m.row(1) << M_SQRT2_2, 0, -M_SQRT2_2;
    model_m.row(2) << M_SQRT2_2, 0, M_SQRT2_2;

    model_yaw = M_PI_2;
    model_pitch = M_PI_4;
    model_roll = 0;

    CheckTwoWays();
}

TEST_F(TwoWayTest, YawAndRollTest) {
    // This matrix sends X to Y.
    // This matrix sends Y to Z.
    // This matrix sends Z to X.
    model_m.row(0) << 0, 0, 1;
    model_m.row(1) << 1, 0, 0;
    model_m.row(2) << 0, 1, 0;

    model_yaw = model_roll = M_PI_2;
    model_pitch = 0;

    CheckTwoWays();
}

TEST_F(TwoWayTest, HalfPitchAndRollTest) {
    // This matrix sends X to (X+Z)/SQRT2.
    // This matrix sends Y to (Z-X)/SQRT2.
    // This matrix sends Z to -Y.
    model_m.row(0) << M_SQRT2_2, -M_SQRT2_2, 0;
    model_m.row(1) << 0, 0, -1;
    model_m.row(2) << M_SQRT2_2, M_SQRT2_2, 0;

    model_yaw = 0;
    model_pitch = M_PI_4;
    model_roll = M_PI_2;

    CheckTwoWays();
}

TEST(RotMatToAnglesTest, RotMatTest_PitchDoesNotChangeImageOfX) {
    float yaw = 1, pitch = 1, roll = 1;
    Eigen::Vector3f x = Eigen::Vector3f::UnitX();
    Eigen::Vector3f y = Eigen::Vector3f::UnitY();

    Eigen::Matrix3f m;

    angles_to_matrix(yaw, 0, 0, m);
    //Eigen::Vector3f x1 = m * x;
    Eigen::Vector3f y1 = m * y;

    angles_to_matrix(yaw, pitch, 0, m);
    Eigen::Vector3f x2 = m * x;
    Eigen::Vector3f y2 = m * y;

    EXPECT_NEAR(y1(0), y2(0), 1e-5)
        << "Adding pitch must not change image of y.";
    EXPECT_NEAR(y1(1), y2(1), 1e-5)
        << "Adding pitch must not change image of y.";
    EXPECT_NEAR(y1(2), y2(2), 1e-5)
        << "Adding pitch must not change image of y.";

    angles_to_matrix(yaw, pitch, roll, m);
    Eigen::Vector3f x3 = m * x;

    EXPECT_NEAR(x2(0), x3(0), 1e-5)
        << "Adding roll must not change image of x.";
    EXPECT_NEAR(x2(1), x3(1), 1e-5)
        << "Adding roll must not change image of x.";
    EXPECT_NEAR(x2(2), x3(2), 1e-5)
        << "Adding roll must not change image of x.";
}

TEST_F(TwoWayTest, FullTest_AnyComboMustConvertAndBack) {

    const float increment = M_PI_2 / 10;

    for (model_yaw = -M_PI * 0.98; model_yaw < M_PI; model_yaw += increment) {
        for (model_pitch = -M_PI_2 * 0.98; model_pitch < M_PI_2;
             model_pitch += increment) {
            for (model_roll = -M_PI_2; model_roll < M_PI_2;
                 model_roll += increment) {
                angles_to_matrix(model_yaw, model_pitch, model_roll, model_m);

                CheckTwoWays();
            }
            std::cerr << ".";
        }
        std::cerr << std::endl;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif // FSG_COMPILING_GTEST_EXECUTABLE

//  LocalWords:  define DEFINES include gtest Eigen Core Geometry fsg
//  LocalWords:  cmath namespace matrixrotationangles OBB ROS ENU Ok
//  LocalWords:  Tait const XY isUnitary TwoWayTest CheckTwoWays EQ
//  LocalWords:  RotMatTest EigenTest CoeffAccessIsMRowColumn YawTest
//  LocalWords:  RotateBookCounterClockwiseQuarterTurnMustYieldYawPi
//  LocalWords:  SQRT PitchTest RollTest YawAndHalfPitchTest FullTest
//  LocalWords:  RotateBookCounterClockwiseEigthTurnMustYieldYawPi
//  LocalWords:  LiftBookPagetopQuarterTurnMustYieldPitchPi argc argv
//  LocalWords:  LiftBookPagetopEighthTurnMustYieldPitchPi endif
//  LocalWords:  OpenBookCoverQuarterTurnMustYieldRollMinusPi
//  LocalWords:  OpenBookCoverEighthTurnMustYieldRollMinusPi
//  LocalWords:  YawAndRollTest HalfPitchAndRollTest
//  LocalWords:  RotMatToAnglesTest PitchDoesNotChangeImageOfX
//  LocalWords:  AnyComboMustConvertAndBack
