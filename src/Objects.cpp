#include "image_processing/Objects.h"

using namespace image_processing;

void Object::update_cloud(PointCloudT current_cloud) {
  // extract features comparing current and previous cloud
  // std::vector<double> feature;
  // features.push_back(feature);
  // update object cloud
  object_cloud = current_cloud;
}
