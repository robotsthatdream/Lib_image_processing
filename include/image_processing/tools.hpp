#ifndef _TOOLS_HPP
#define _TOOLS_HPP

#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/search/kdtree.h>

#include "pcl_types.h"
#include "SupervoxelSet.h"

namespace image_processing{
namespace tools{
//class tools{

//public:
    //TO DO : Templatize
    //template<typename point>
    bool extract_convex_hull(pcl::PointCloud<PointT>::ConstPtr cloud,  std::vector<Eigen::Vector3d>& vertex_list);

    /** \brief Convert a RGB tuple to an HSV one.
 * \param[in] r the input Red component
 * \param[in] g the input Green component
 * \param[in] b the input Blue component
 * \param[out] fh the output Hue component
 * \param[out] fs the output Saturation component
 * \param[out] fv the output Value component
 */
    void rgb2hsv (int r, int g, int b, float& fh, float& fs, float& fv);

    /**
 * @brief convert a pcl::PointCloud<PointXYZRGB> to pcl::PointCloud<PointXYZHSV>
 * @param input cloud
 * @param output cloud
 */
    void cloudRGB2HSV(const PointCloudT::Ptr input, PointCloudHSV::Ptr output);

    void rgb2Lab(int r, int g, int b, float& L, float& a, float& b2);



}//tools
}//image_processing

#endif //_TOOLS_HPP
