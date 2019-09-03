#ifndef NUMBER_TYPE_HPP

#define USE_DOUBLE 1

#ifdef USE_DOUBLE
#define FNUM_TYPE double
#define FNUM_LITERAL(number) number

#define WITH_SUFFIX_fx(symbolname) symbolname
#define WITH_SUFFIX_fd(symbolname) symbolname##d
#define WITH_SUFFIX_xd(symbolname) symbolname##d

#define WITH_INFIX_fx(prefix, suffix) prefix ## suffix
#define WITH_INFIX_fd(prefix, suffix) prefix ## d ## suffix
#define WITH_INFIX_xd(prefix, suffix) prefix ## d ## suffix

#else

#define FNUM_TYPE float
#define FNUM_LITERAL(number) number##f

#define WITH_SUFFIX_fx(symbolname) symbolname##f
#define WITH_SUFFIX_fd(symbolname) symbolname##f
#define WITH_SUFFIX_xd(symbolname) symbolname

#define WITH_INFIX_fx(prefix, suffix) prefix ## f ## suffix
#define WITH_INFIX_fd(prefix, suffix) prefix ## f ## suffix
#define WITH_INFIX_xd(prefix, suffix) prefix ## suffix

#endif

#pragma GCC diagnostic warning "-Wdouble-promotion"
#pragma GCC diagnostic warning "-Wfloat-equal"
#pragma GCC diagnostic warning "-Wfloat-conversion"
#pragma GCC diagnostic warning "-Wconversion"

static constexpr FNUM_TYPE sg_pi = (FNUM_TYPE) M_PI;
static constexpr FNUM_TYPE sg_pi_2 = (FNUM_TYPE) M_PI_2;
static constexpr FNUM_TYPE sg_pi_4 = (FNUM_TYPE) M_PI_4;
static constexpr FNUM_TYPE sg_0 = FNUM_LITERAL(0.0);
static constexpr FNUM_TYPE sg_1 = FNUM_LITERAL(1.0);
static constexpr FNUM_TYPE sg_2 = FNUM_LITERAL(2.0);
static constexpr FNUM_TYPE sg_half = FNUM_LITERAL(0.5);

#define VECTOR3 Eigen::WITH_SUFFIX_fd(Vector3)
#define MATRIX3 Eigen::WITH_SUFFIX_fd(Matrix3)
#define VECTOR4 Eigen::WITH_SUFFIX_fd(Vector4)
#define MATRIX4 Eigen::WITH_SUFFIX_fd(Matrix4)
#define VECTORX Eigen::WITH_SUFFIX_fd(VectorX)

#endif /* NUMBER_TYPE_HPP */
