namespace fsg {
namespace matrixrotationangles {

    
/*

  ===== Definition of rotational matrix =====

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
                          float &roll);
    
    void angles_to_matrix(const float &yaw, const float &pitch, const float &roll,
                          Eigen::Matrix3f &m);

}
}
