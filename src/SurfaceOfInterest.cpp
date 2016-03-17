#include <SurfaceOfInterest.h>

using namespace image_processing;

void SurfaceOfInterest::find_soi(const PointCloudXYZ::Ptr key_pts){
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZ>());

        PointCloudT centr;
        std::map<int, uint32_t> centroids_label;
        getCentroidCloud(centr, centroids_label);
        PointCloudXYZ::Ptr centroids(new PointCloudXYZ);
        for(int i = 0; i < centr.size(); i++){
            pcl::PointXYZ pt;
            pt.x = centr.points[i].x;
            pt.y = centr.points[i].y;
            pt.z = centr.points[i].z;
            centroids->push_back(pt);
        }

        tree->setInputCloud(centroids);

        std::vector<int> nn_indices(1);
        std::vector<float> nn_distance(1);

        std::map<uint32_t,uint32_t> result;

        for(int i = 0; i < key_pts->size(); i++){
            if(!pcl::isFinite(key_pts->points[i]))
                continue;

            if(!tree->nearestKSearch(key_pts->points[i],1,nn_indices,nn_distance)){
                PCL_WARN ("No neighbor found for point %lu (%f %f %f)!\n",
                          i, key_pts->points[i].x, key_pts->points[i].y, key_pts->points[i].z);
                continue;
            }
            if(nn_distance[0] < 0.05)
                result.emplace(centroids_label[nn_indices[0]],centroids_label[nn_indices[0]]);
        }

        for(auto it = result.begin(); it != result.end(); it++){
            _labels.push_back(it->first);
            _weights.emplace(it->first,1.);
        }
}

void SurfaceOfInterest::reduce_to_soi(){
    for(int i = 0; i < _labels.size(); i++){
        remove(_labels[i]);
    }
    consolidate();
}

void SurfaceOfInterest::choice_of_soi(pcl::Supervoxel<PointT> &supervoxel, uint32_t &lbl){

    //*build the distribution from weights
    std::map<float,uint32_t> soi_dist;
    float val = 0.f;
    for(auto it = _weights.begin(); it != _weights.end(); it++){
        val+=it->second/((float)_labels.size());
        soi_dist.emplace(val,it->first);
    }
    //*/

    //*choice of a supervoxel
    boost::random::uniform_real_distribution<> dist(0.,1.);

    float choice = dist(_gen);
    lbl = soi_dist.lower_bound(choice)->second;
    supervoxel = *_supervoxels[lbl];
    //*/
}
