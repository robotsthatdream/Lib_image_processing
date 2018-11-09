#define _USE_MATH_DEFINES
#include "gtest/gtest.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace robotsthatdream {

/*

  ===== Defition of rotational matrix =====

  Rotational_matrix_OBB M is defined by: its *columns* are the
  components of the major (e1), middle (e2), minor (e3) axes.

  Properties:

  M*[1 0 0] -> e1
  M*[0 1 0] -> e2
  M*[0 0 1] -> e3

  Thus, the rotational_matrix sends X to e1, Y to e2, Z to e3.

  We say rotation matrix M rotates from frame F to frame G.

  In other words, the rotational matrix M, for any point P in space
  whose coordinates are expressed p1p2p3 in frame F, computes the
  coordinates in frame F of the point Q which has coordinates p1p2p3
  in frame G.

  Okay.

*/

/*

  ===== Definition of coordinate system: axis convention =====

  First, we have to choose a coordinate system convention.

  We choose the axis orientation convention used by the ROS project,
  relative to the body (not the camera) of the robot.
  http://www.ros.org/reps/rep-0103.html#axis-orientation

  - x forward
  - y left
  - z up

           Z

           |
         x |
          \|
    Y -----O

  To simplify things, we imagine the robot facing east.  In this case,
  body-based axes orientation matches the ROS conventions:

  > For short-range Cartesian representations of geographic locations,
  > use the east north up [5] (ENU) convention:
  >
  > - X east
  > - Y north
  > - Z up

  We'll say the "neutral" orientation of the robot is standing facing
  east.

*/

/*

  ===== Concrete illustration of axis convention =====

  Let's make things more concrete with a familiar object: a book.

  Consider a book with rigid cover in 3D.

  - Strongest moment is page height (longest dimension),

  - second moment is page width,

  - third moment is book thickness (shortest dimension).


  We'll define a "neutral" position of the book.

  - The book is lying closed on a horizontal table.

  - The top of the pages is towards east.

  - The right of the pages is towards south.

  - From front cover to back cover (drilling through the book), is
    going downward.

  This "neutral" book position is "neutral", that is rotation matrix
  is identity, rotation angles are zero.


  - From bottom of pages to top of pages, x increases.

  - From left of cover page to right of cover page, y decreases.

  - From front cover to back cover (drilling through the book), z
    decreases.

*/

/*

  ===== Definition of angle convention =====

  We have to choose an angle convention,

  ROS says:

  > fixed axis roll, pitch, yaw about X, Y, Z axes respectively
  >
  > * No ambiguity on order
  > * Used for angular velocities

  I understand it in a way that matches the Tait-Bryan convention (as
  explained on https://en.wikipedia.org/wiki/Euler_angles ).

  - First angle provides general orientation of the book as viewed
    from above.

  - Second angle corresponds to lifting the top of the page towards
    you.

  - Third angle corresponds to opening the book cover.

  These angles correspond also to "natural" ways to describe aircraft
  orientation and orientation of a camera on a tripod.


  You can now refer to illustration on
  https://en.wikipedia.org/wiki/Euler_angles#/media/File:Taitbrianzyx.svg
  oh forget that, no don't even look at it, it's awful. ;-)

  Now, if you turn the book, keeping it still lying on the table, the
  angle "yaw" will reflect that.  "Yaw" will increase from 0 to pi
  then -pi to 0 as you rotate the book counter-clockwise.

  Whatever the value of "yaw", if you lift the top of the pages, keeping
  the bottom of the pages on the table, you will increase the second
  angle, "pitch".  "Pitch" can go from -pi/2 to +pi/2.

  Whatever the value of "yaw" and "pitch", you can open the book.  The
  motion of the cover page is a decrease of "roll".  "Roll" can go from -pi/2
  to +pi/2.

*/

/*

  We need a way to transform once a rotation matrix into angles "yaw",
  "pitch", "roll": to deduce angles from the matrix found by
  pcl::MomentOfInertiaEstimation.

  Converting from angles to rotation matrix: [c++ - How to calculate
  the angle from rotation matrix - Stack
  Overflow](https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
  "c++ - How to calculate the angle from rotation matrix - Stack
  Overflow")

  We need a way, given angles "yaw", "pitch", "roll", to rotate a point
  cloud.  We'll follow
  http://pointclouds.org/documentation/tutorials/matrix_transform.php

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

    std::cerr << "m_without_yaw" << std::endl << m_without_yaw << std::endl;

    // pitch = atan2(m(2, 0), hypot( m(2, 2) ) );

    /* Now we want pitch.  We know the matrix no longer has yaw.

       It's the angle from X to the projection of its image on the XZ plane.

       In other words, it's the atan2 (component of image of X on Z, component
       of image of X on X).
    */

    pitch = atan2(m_without_yaw(2, 0), m_without_yaw(0, 0));

    const Eigen::Matrix3f m_without_yaw_nor_pitch =
        Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) * m_without_yaw;

    //std::cerr << "m_without_yaw_nor_pitch" << std::endl << m_without_yaw_nor_pitch << std::endl;


    /* Ok, we're nearly there.  We know the matrix is only roll now.

       It's the angle from Y to the projection of its image on the YZ plane.

       In other words, it's the atan2 (component of image of Y on Z, component
       of image of Y on Y).

     */

    roll = atan2(m_without_yaw_nor_pitch(2, 1), m_without_yaw_nor_pitch(1, 1));
}

/*

  Eigen offers ways to "easily" convert from angles to a rotation
  matrix:

  > Combined with MatrixBase::Unit{X,Y,Z}, AngleAxis can be used to
  easily mimic Euler-angles. Here is an example:
  > Matrix3f m;
  > m = AngleAxisf(0.25*M_PI, Vector3f::UnitX())
  >   * AngleAxisf(0.5*M_PI,  Vector3f::UnitY())
  >   * AngleAxisf(0.33*M_PI, Vector3f::UnitZ());
  > cout << m << endl << "is unitary: " << m.isUnitary() << endl;

  http://eigen.tuxfamily.org/dox/classEigen_1_1Matrix.html#a2f6bdcb76b48999cb9135b828bba4e7d

  TODO find correct convention.

*/

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

// The fixture for testing class RotMatToAnglesTest.
class RotMatTest : public ::testing::Test {
  protected:
    // You can remove any or all of the following functions if its body
    // is empty.

    Eigen::Matrix3f m;
    const Eigen::Matrix3f identity = Eigen::MatrixXf::Identity(3, 3);
    float yaw, pitch, roll;

    RotMatTest() {
        // std::cerr << std::endl
        //           <<
        //           "========================================================="
        //           << std::endl;
        // You can do set-up work for each test here.
    }

    ~RotMatTest() override {
        // You can do clean-up work that doesn't throw exceptions here.
        // std::cerr <<
        // "========================================================="
        //           << std::endl
        //           << std::endl;
    }
};

class RotMatToAnglesTest : public RotMatTest {
  protected:
    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp() override {}

    void TearDown() override {
        std::cerr << " yaw=" << yaw << " pitch=" << pitch << " roll=" << roll
                  << std::endl;
    }

    // Objects declared here can be used by all tests in the test case for
    // RotMatToAnglesTest.
};

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

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp() override {}

    void TearDown() override {}

    void CheckTwoWays() {
        std::cerr << model_m << std::endl
                  << " model_yaw=" << model_yaw
                  << " model_pitch=" << model_pitch
                  << " model_roll=" << model_roll << std::endl;

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

    // Objects declared here can be used by all tests in the test case for
    // RotMatToAnglesTest.
};

TEST_F(RotMatTest, EigenTest_CoeffAccessIsMRowColumn) {
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

TEST_F(RotMatToAnglesTest, RotMatTest_PitchDoesNotChangeImageOfX) {
    float yaw = 1, pitch = 1, roll = 1;
    Eigen::Vector3f x = Eigen::Vector3f::UnitX();
    Eigen::Vector3f y = Eigen::Vector3f::UnitY();

    angles_to_matrix(yaw, 0, 0, m);
    Eigen::Vector3f x1 = m * x;
    Eigen::Vector3f y1 = m * y;

    angles_to_matrix(yaw, pitch, 0, m);
    Eigen::Vector3f x2 = m * x;
    Eigen::Vector3f y2 = m * x;

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

TEST_F(RotMatToAnglesTest, FullTest_AnyComboMustConvertAndBack) {

    const float increment = 1;

    for (float orig_yaw = 0; orig_yaw < M_PI; orig_yaw += increment) {
        for (float orig_pitch = 0; orig_pitch < 1.5; orig_pitch += increment) {
            for (float orig_roll = 0; orig_roll < 1.5; orig_roll += increment) {

                angles_to_matrix(orig_yaw, orig_pitch, orig_roll, m);

                std::cerr << m << std::endl;
                matrix_to_angles(m, yaw, pitch, roll);

                EXPECT_NEAR(yaw, orig_yaw, 1e-5);
                EXPECT_NEAR(pitch, orig_pitch, 1e-5);
                EXPECT_NEAR(roll, orig_roll, 1e-5);
            }
        }
    }
}
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
