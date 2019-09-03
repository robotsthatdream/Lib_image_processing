#define _USE_MATH_DEFINES
#include "test_rotation.hpp"
#include "number_type.hpp"
#include "gtest/gtest.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace fsg
{
namespace matrixrotationangles
{

/* See documentation for the following topics in test_rotation.hpp:

   - Definition of rotational matrix

   - Definition of coordinate system: axis convention

   - Concrete illustration of axis convention

   - Definition of angle convention

 */

void matrix_to_angles(const MATRIX3 &m, FNUM_TYPE &yaw, FNUM_TYPE &pitch,
                      FNUM_TYPE &roll)
{
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
    yaw = WITH_SUFFIX_fx(atan2)(m(sg_1, sg_0), m(sg_0, sg_0));

    const MATRIX3 m_without_yaw =
        Eigen::WITH_SUFFIX_fd(AngleAxis)(-yaw, VECTOR3::UnitZ()) * m;

    // std::cerr << "m_without_yaw" << std::endl << m_without_yaw << std::endl;

    /* Now we want pitch.  We know the matrix no longer has yaw.

       It's the angle from X to the projection of its image on the XZ plane.

       In other words, it's the atan2 (component of image of X on Z, component
       of image of X on X).
    */

    pitch = WITH_SUFFIX_fx(atan2)(
        m_without_yaw(FNUM_LITERAL(2.0), FNUM_LITERAL(0.0)),
        m_without_yaw(FNUM_LITERAL(0.0), FNUM_LITERAL(0.0)));

    const MATRIX3 m_without_yaw_nor_pitch =
        Eigen::WITH_SUFFIX_fd(AngleAxis)(pitch, VECTOR3::UnitY()) *
        m_without_yaw;

    // std::cerr << "m_without_yaw_nor_pitch" << std::endl <<
    // m_without_yaw_nor_pitch << std::endl;

    /* Ok, we're nearly there.  We know the matrix is only roll now.

       It's the angle from Y to the projection of its image on the YZ plane.

       In other words, it's the atan2 (component of image of Y on Z, component
       of image of Y on Y).

     */

    roll = WITH_SUFFIX_fx(atan2)(
        m_without_yaw_nor_pitch(FNUM_LITERAL(2.0), FNUM_LITERAL(1.0)),
        m_without_yaw_nor_pitch(FNUM_LITERAL(1.0), FNUM_LITERAL(1.0)));
}

void angles_to_matrix(const FNUM_TYPE &yaw, const FNUM_TYPE &pitch,
                      const FNUM_TYPE &roll, MATRIX3 &m)
{
    // std::cerr << __PRETTY_FUNCTION__ << " yaw=" << yaw << " pitch=" << pitch
    //           << " roll=" << roll << std::endl;
    m = Eigen::WITH_SUFFIX_fd(AngleAxis)(yaw, VECTOR3::UnitZ()) *
        Eigen::WITH_SUFFIX_fd(AngleAxis)(-pitch, VECTOR3::UnitY()) *
        Eigen::WITH_SUFFIX_fd(AngleAxis)(roll, VECTOR3::UnitX());
    // std::cerr << m << std::endl << "is unitary: " << m.isUnitary() <<
    // std::endl;
}
} // namespace matrixrotationangles
} // namespace fsg

// =====================================================================
// Test below
// =====================================================================

#ifdef FSG_COMPILING_GTEST_EXECUTABLE

using namespace fsg::matrixrotationangles;

void expect_identical_3x3_matrices(const MATRIX3 &m1, const MATRIX3 &m2,
                                   const FNUM_TYPE epsilon = FNUM_LITERAL(1e-5))
{
    for (int row = 0; row <= 2; row++)
    {
        for (int col = 0; col <= 2; col++)
        {
            EXPECT_NEAR(m1(row, col), m2(row, col), epsilon) << "row=" << row
                                                             << ", col=" << col;
        }
    }
}

bool check_if_unitary(const MATRIX3 &m)
{
    for (int i = 0; i < 3; ++i)
    {
        std::cerr << "norm(col(" << i << ")) = " << m.col(i).squaredNorm()
                  << std::endl;
        for (int j = 0; j < i; ++j)
        {
            std::cerr << "(col(" << i << ")).(col(" << j
                      << ")) = " << m.col(i).dot(m.col(j)) << std::endl;
        }
    }
    return true;
}

class TwoWayTest : public ::testing::Test
{
  protected:
    MATRIX3 model_m;
    FNUM_TYPE model_yaw, model_pitch, model_roll;

    void CheckTwoWays()
    {
        // std::cerr << model_m << std::endl
        //           << " model_yaw=" << model_yaw
        //           << " model_pitch=" << model_pitch
        //           << " model_roll=" << model_roll << std::endl;

        if (!model_m.isUnitary())
        {
            check_if_unitary(model_m);
        }

        ASSERT_TRUE(model_m.isUnitary());

        FNUM_TYPE yaw_computed, pitch_computed, roll_computed;
        matrix_to_angles(model_m, yaw_computed, pitch_computed, roll_computed);

        EXPECT_NEAR(model_yaw, yaw_computed, FNUM_LITERAL(1e-5));
        EXPECT_NEAR(model_pitch, pitch_computed, FNUM_LITERAL(1e-5));
        EXPECT_NEAR(model_roll, roll_computed, FNUM_LITERAL(1e-5));

        MATRIX3 m_computed;
        angles_to_matrix(model_yaw, model_pitch, model_roll, m_computed);
        expect_identical_3x3_matrices(model_m, m_computed);
    }
};

TEST(RotMatTest, EigenTest_CoeffAccessIsMRowColumn)
{
    MATRIX3 m;

    m.row(0) << FNUM_LITERAL(1.0), FNUM_LITERAL(2.0), FNUM_LITERAL(3.0);
    m.row(1) << FNUM_LITERAL(4.0), FNUM_LITERAL(5.0), FNUM_LITERAL(6.0);
    m.row(2) << FNUM_LITERAL(7.0), FNUM_LITERAL(8.0), FNUM_LITERAL(9.0);

    EXPECT_EQ(m(0, 0), FNUM_LITERAL(1.0));
    EXPECT_EQ(m(1, FNUM_LITERAL(0.0)), 4);
    EXPECT_EQ(m(FNUM_LITERAL(2.0), FNUM_LITERAL(0.0)), 7);
    EXPECT_EQ(m(FNUM_LITERAL(2.0), 1), 8);
    EXPECT_EQ(m(FNUM_LITERAL(2.0), 2), 9);
}

TEST_F(TwoWayTest, Identity)
{
    model_m.row(0) << 1, FNUM_LITERAL(0.0), FNUM_LITERAL(0.0);
    model_m.row(1) << FNUM_LITERAL(0.0), 1, FNUM_LITERAL(0.0);
    model_m.row(2) << FNUM_LITERAL(0.0), FNUM_LITERAL(0.0), 1;

    model_yaw = model_pitch = model_roll = FNUM_LITERAL(0.0);

    CheckTwoWays();
}

TEST_F(TwoWayTest, YawTest_RotateBookCounterClockwiseQuarterTurnMustYieldYawPi2)
{
    // This matrix sends X to Y.
    // This matrix sends Y to -X.
    // This matrix sends Z to Z.
    model_m.row(0) << FNUM_LITERAL(0.0), -1, FNUM_LITERAL(0.0);
    model_m.row(1) << 1, FNUM_LITERAL(0.0), FNUM_LITERAL(0.0);
    model_m.row(2) << FNUM_LITERAL(0.0), FNUM_LITERAL(0.0), 1;

    model_yaw = sg_pi_2;
    model_pitch = model_roll = FNUM_LITERAL(0.0);

    CheckTwoWays();
}

#define M_SQRT2_2 (((FNUM_TYPE)M_SQRT2) / FNUM_LITERAL(2.0))

TEST_F(TwoWayTest, YawTest_RotateBookCounterClockwiseEigthTurnMustYieldYawPi4)
{
    // This matrix sends X to Y.
    // This matrix sends Y to -X.
    // This matrix sends Z to Z.
    model_m.row(0) << M_SQRT2_2, -M_SQRT2_2, FNUM_LITERAL(0.0);
    model_m.row(1) << M_SQRT2_2, M_SQRT2_2, FNUM_LITERAL(0.0);
    model_m.row(2) << FNUM_LITERAL(0.0), FNUM_LITERAL(0.0), 1;

    model_yaw = sg_pi_4;
    model_pitch = model_roll = FNUM_LITERAL(0.0);

    CheckTwoWays();
}

TEST_F(TwoWayTest, PitchTest_LiftBookPagetopQuarterTurnMustYieldPitchPi2)
{
    // This matrix sends X to Z.
    // This matrix sends Y to Y.
    // This matrix sends Z to -X.
    model_m.row(0) << FNUM_LITERAL(0.0), FNUM_LITERAL(0.0), -1;
    model_m.row(1) << FNUM_LITERAL(0.0), 1, FNUM_LITERAL(0.0);
    model_m.row(2) << 1, FNUM_LITERAL(0.0), FNUM_LITERAL(0.0);

    model_yaw = model_roll = FNUM_LITERAL(0.0);
    model_pitch = sg_pi_2;

    CheckTwoWays();
}

TEST_F(TwoWayTest, PitchTest_LiftBookPagetopEighthTurnMustYieldPitchPi4)
{
    // This matrix sends X to Z.
    // This matrix sends Y to Y.
    // This matrix sends Z to -X.
    model_m.row(0) << M_SQRT2_2, FNUM_LITERAL(0.0), -M_SQRT2_2;
    model_m.row(1) << FNUM_LITERAL(0.0), 1, FNUM_LITERAL(0.0);
    model_m.row(2) << M_SQRT2_2, FNUM_LITERAL(0.0), M_SQRT2_2;

    model_yaw = model_roll = FNUM_LITERAL(0.0);
    model_pitch = sg_pi_4;

    CheckTwoWays();
}

TEST_F(TwoWayTest, RollTest_OpenBookCoverQuarterTurnMustYieldRollMinusPi2)
{
    // This matrix sends X to X.
    // This matrix sends Y to -Z.
    // This matrix sends Z to Y.
    model_m.row(0) << 1, FNUM_LITERAL(0.0), FNUM_LITERAL(0.0);
    model_m.row(1) << FNUM_LITERAL(0.0), FNUM_LITERAL(0.0), 1;
    model_m.row(2) << FNUM_LITERAL(0.0), -1, FNUM_LITERAL(0.0);

    model_yaw = model_pitch = FNUM_LITERAL(0.0);
    model_roll = -sg_pi_2;

    CheckTwoWays();
}

TEST_F(TwoWayTest, RollTest_OpenBookCoverEighthTurnMustYieldRollMinusPi4)
{
    // This matrix sends X to X.
    // This matrix sends Y to -Z.
    // This matrix sends Z to Y.
    model_m.row(0) << 1, FNUM_LITERAL(0.0), FNUM_LITERAL(0.0);
    model_m.row(1) << FNUM_LITERAL(0.0), M_SQRT2_2, M_SQRT2_2;
    model_m.row(2) << FNUM_LITERAL(0.0), -M_SQRT2_2, M_SQRT2_2;

    model_yaw = model_pitch = FNUM_LITERAL(0.0);
    model_roll = sg_pi_4;

    CheckTwoWays();
}

TEST_F(TwoWayTest, YawAndHalfPitchTest)
{
    // This matrix sends X to (Y+Z)/SQRT2.
    // This matrix sends Y to -X.
    // This matrix sends Z to (-Y+Z)/SQRT2.
    model_m.row(0) << FNUM_LITERAL(0.0), -1, FNUM_LITERAL(0.0);
    model_m.row(1) << M_SQRT2_2, FNUM_LITERAL(0.0), -M_SQRT2_2;
    model_m.row(2) << M_SQRT2_2, FNUM_LITERAL(0.0), M_SQRT2_2;

    model_yaw = sg_pi_2;
    model_pitch = sg_pi_4;
    model_roll = FNUM_LITERAL(0.0);

    CheckTwoWays();
}

TEST_F(TwoWayTest, YawAndRollTest)
{
    // This matrix sends X to Y.
    // This matrix sends Y to Z.
    // This matrix sends Z to X.
    model_m.row(0) << FNUM_LITERAL(0.0), FNUM_LITERAL(0.0), 1;
    model_m.row(1) << 1, FNUM_LITERAL(0.0), FNUM_LITERAL(0.0);
    model_m.row(2) << FNUM_LITERAL(0.0), 1, FNUM_LITERAL(0.0);

    model_yaw = model_roll = sg_pi_2;
    model_pitch = FNUM_LITERAL(0.0);

    CheckTwoWays();
}

TEST_F(TwoWayTest, HalfPitchAndRollTest)
{
    // This matrix sends X to (X+Z)/SQRT2.
    // This matrix sends Y to (Z-X)/SQRT2.
    // This matrix sends Z to -Y.
    model_m.row(0) << M_SQRT2_2, -M_SQRT2_2, FNUM_LITERAL(0.0);
    model_m.row(1) << FNUM_LITERAL(0.0), FNUM_LITERAL(0.0), -1;
    model_m.row(2) << M_SQRT2_2, M_SQRT2_2, FNUM_LITERAL(0.0);

    model_yaw = FNUM_LITERAL(0.0);
    model_pitch = sg_pi_4;
    model_roll = sg_pi_2;

    CheckTwoWays();
}

TEST(RotMatToAnglesTest, RotMatTest_PitchDoesNotChangeImageOfX)
{
    FNUM_TYPE yaw = 1, pitch = 1, roll = 1;
    VECTOR3 x = VECTOR3::UnitX();
    VECTOR3 y = VECTOR3::UnitY();

    MATRIX3 m;

    angles_to_matrix(yaw, FNUM_LITERAL(0.0), FNUM_LITERAL(0.0), m);
    // VECTOR3 x1 = m * x;
    VECTOR3 y1 = m * y;

    angles_to_matrix(yaw, pitch, FNUM_LITERAL(0.0), m);
    VECTOR3 x2 = m * x;
    VECTOR3 y2 = m * y;

    EXPECT_NEAR(y1(FNUM_LITERAL(0.0)), y2(FNUM_LITERAL(0.0)), 1e-5)
        << "Adding pitch must not change image of y.";
    EXPECT_NEAR(y1(1), y2(1), 1e-5)
        << "Adding pitch must not change image of y.";
    EXPECT_NEAR(y1(2), y2(2), 1e-5)
        << "Adding pitch must not change image of y.";

    angles_to_matrix(yaw, pitch, roll, m);
    VECTOR3 x3 = m * x;

    EXPECT_NEAR(x2(FNUM_LITERAL(0.0)), x3(FNUM_LITERAL(0.0)), 1e-5)
        << "Adding roll must not change image of x.";
    EXPECT_NEAR(x2(1), x3(1), 1e-5)
        << "Adding roll must not change image of x.";
    EXPECT_NEAR(x2(2), x3(2), 1e-5)
        << "Adding roll must not change image of x.";
}

TEST_F(TwoWayTest, FullTest_AnyComboMustConvertAndBack)
{

    const FNUM_TYPE increment = (sg_pi_2) / (FNUM_LITERAL(10.0));

    for (model_yaw = (-sg_pi) * (FNUM_LITERAL(0.98)); model_yaw < sg_pi;
         model_yaw += increment)
    {
        for (model_pitch = (-sg_pi_2) * (FNUM_LITERAL(0.98));
             model_pitch < sg_pi_2; model_pitch += increment)
        {
            for (model_roll = -sg_pi_2; model_roll < sg_pi_2;
                 model_roll += increment)
            {
                angles_to_matrix(model_yaw, model_pitch, model_roll, model_m);

                CheckTwoWays();
            }
            std::cerr << ".";
        }
        std::cerr << std::endl;
    }
}

int main(int argc, char **argv)
{
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
