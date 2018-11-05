#include "gtest/gtest.h"
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace robotsthatdream {

/* rotational_matrix_OBB M is defined by: its columns are the
 * components of the major (e1), middle (e2), minor (e3) axes.

 Properties:

 M*[1 0 0] -> e1
 M*[0 1 0] -> e2
 M*[0 0 1] -> e3

 This, the rotational_matrix sends X to e1, Y to e2, Z to e3.

 So, the rotational matrix sends any point in space to the point that
 would have same coordinates in the object's reference frame.

 Okay.
*/

/*
  We have to choose an angle convention,
  cf. https://en.wikipedia.org/wiki/Euler_angles .

  We choose Tait-Bryan with strongest moment aligned with X axis, and
  weakest with Z axis.

  Why ? It is intuitive.  Consider a book with rigid cover in 3D.
  Strongest moment is page height (longest dimension), second page
  width, third book thickness (shortest dimension).

  - First angle provides general orientation (with respect to north)
  viewed from above.

  - Second angle corresponds to lifting the top of the page towards
  you.

  - Third angle corresponds to opening the book cover.

  These angles correspond also to "natural" ways to describe aircraft
  orientation and orientation of a camera on a tripod.
*/

/* Now we need a coordinate system convention.

   We choose the already existing convention of camera-centered
   coordinates:

   * x right
   * y down
   * z forward

   O----- X
   |\
   | z
   |

   Y

   Let's elaborate the book with rigid cover, with axes.

   Okay, so imagine a book with rigid cover, lying on a horizontal
   table, oriented like you orient a book to read, and the robot is
   looking horizontally.

   * All angles are zero.

   * From bottom of pages to top of pages, z increases.

   * From left of cover page to right of cover page, x increases.

   * From front cover to back cover (drilling through the book), y
   increases.

   You can now refer to illustration on
   https://en.wikipedia.org/wiki/Euler_angles#/media/File:Taitbrianzyx.svg

   Now, if you turn the book, keeping it still lying on the table, the
   first angle psi will reflect that.  Psi will increase from 0 to 2pi
   as you rotate the book counter-clockwise.

   Whatever the value of psi, if you lift the top of the pages,
   keeping the bottom of the pages on the table, you will increase the
   second angle, theta.  Theta can go from -pi to +pi.

   Whatever the value of psi and theta, you can open the book.  The
   motion of the cover page is a decrease of phi.  Phi can go from -pi
   to +pi.

*/

/*

  We need a way to transform once a rotation matrix into angles psi,
  theta, phi: to deduce angles from the matrix found by
  pcl::MomentOfInertiaEstimation.

  Converting from angles to rotation matrix: [c++ - How to calculate
  the angle from rotation matrix - Stack
  Overflow](https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
  "c++ - How to calculate the angle from rotation matrix - Stack
  Overflow")

  We need a way, given angles psi, theta, phi, to rotate a point
  cloud.  We'll follow
  http://pointclouds.org/documentation/tutorials/matrix_transform.php

*/

void matrix_to_angles(const Eigen::Matrix3f &m, float &psi, float &theta,
                      float &phi) {
    /* Ok, so how do we compute our angles?

       Let's call our rotation matrix M[l,c] = [ e1 e2 e3 ]

       First angle psi depends only on e1 (vector of the major axis of the
       object/book).
    */
    psi = atan2(-m(2, 0), m(0, 0));
    theta = 0;
    phi = 0;
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

// The fixture for testing class DreamRotationMatrixAngles.
class RotMatToAnglesTest : public ::testing::Test {
  protected:
    // You can remove any or all of the following functions if its body
    // is empty.

    Eigen::Matrix3f m;
    float psi, theta, phi;

    RotMatToAnglesTest() {
        // You can do set-up work for each test here.
    }

    ~RotMatToAnglesTest() override {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp() override {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override {
        std::cerr << m << std::endl;
        std::cerr << "psi=" << psi << " theta=" << theta << " phi=" << phi
                  << std::endl;
    }

    // Objects declared here can be used by all tests in the test case for
    // DreamRotationMatrixAngles.
};

TEST_F(RotMatToAnglesTest, Identity) {
    m << 1, 0, 0, 0, 1, 0, 0, 0, 1;

    matrix_to_angles(m, psi, theta, phi);

    EXPECT_NEAR(psi, 0, 1e-5);
    EXPECT_NEAR(theta, 0, 1e-5);
    EXPECT_NEAR(phi, 0, 1e-5);
}
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
