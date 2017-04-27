#include "image_processing/Objects.h"

using namespace image_processing;

void Blob::add_child(Blob& blob)
{
  children.push_back(blob);
}


void Objects::add(PointCloudT blob_cloud)
{
  Blob blob(blob_cloud);
  objects.push_back(blob);
}

void Objects::remove(std::vector<Blob>::iterator position)
{
  objects.erase(position);
}

void Objects::merge(std::vector<Blob>::iterator pos1, std::vector<Blob>::iterator pos2,
  PointCloudT parent_cloud)
{
  Blob parent(parent_cloud);
  parent.add_child(*pos1);
  parent.add_child(*pos2);
  if (pos2 < pos1) {
    objects.erase(pos1);
    objects.erase(pos2);
  }
  else {
    objects.erase(pos2);
    objects.erase(pos1);
  }
  objects.push_back(parent);
}

void Objects::split(std::vector<Blob>::iterator pos)
{
  for (auto child : pos->children)
  {
    objects.push_back(child);
  }
  objects.erase(pos);
}
