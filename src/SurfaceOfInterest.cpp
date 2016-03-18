#include <SurfaceOfInterest.h>

using namespace image_processing;

void SurfaceOfInterest::find_soi(const PointCloudXYZ::Ptr key_pts){
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZ>());

        PointCloudT centr;
        std::map<int, uint32_t> centroids_label;
        _labels.clear();
        _labels_no_soi.clear();
        _weights.clear();
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
        std::map<uint32_t,uint32_t> no_result;

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
            else
                no_result.emplace(centroids_label[nn_indices[0]],centroids_label[nn_indices[0]]);
        }


        for(auto it = no_result.begin(); it != no_result.end(); it++)
            _labels_no_soi.push_back(it->first);

        for(auto it = result.begin(); it != result.end(); it++){
            _labels.push_back(it->first);
            _weights.emplace(it->first,1.);
        }
        assert(_labels.size() == _weights.size());
}

void SurfaceOfInterest::reduce_to_soi(){
    for(int i = 0; i < _labels_no_soi.size(); i++){
        remove(_labels_no_soi[i]);
    }
    consolidate();
}

void SurfaceOfInterest::choice_of_soi(pcl::Supervoxel<PointT> &supervoxel, uint32_t &lbl){

    //*build the distribution from weights
    std::map<float,uint32_t> soi_dist;
    float val = 0.f;
    for(auto it = _weights.begin(); it != _weights.end(); it++){
        val+=it->second/((float)_weights.size());
        soi_dist.emplace(val,it->first);
    }
    //*/

    //*choice of a supervoxel
    boost::random::uniform_real_distribution<> dist(0.,1.);

    float choice = dist(_gen);
    lbl = soi_dist.lower_bound(choice)->second;
    supervoxel = *(_supervoxels[lbl]);
    //*/
}
