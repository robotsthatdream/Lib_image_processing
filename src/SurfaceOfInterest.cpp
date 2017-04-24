#include "image_processing/SurfaceOfInterest.h"
#include "eigen3/Eigen/Core"
#include "opencv2/ocl/ocl.hpp"

using namespace image_processing;

//NAIVE POLICY
bool SurfaceOfInterest::generate(workspace_t &workspace){
    if(!computeSupervoxel(workspace))
        return false;

    init_weights("random",2);
    return true;
}

//KEYPOINTS POLICY
bool SurfaceOfInterest::generate(const PointCloudXYZ::Ptr key_pts, workspace_t &workspace){
    if(!computeSupervoxel(workspace))
        return false;

    init_weights("keyPts",2,0.);
    find_soi(key_pts);
    return true;
}

//EXPERT POLICY
bool SurfaceOfInterest::generate(const PointCloudT::Ptr background, workspace_t &workspace){
    delete_background(background);
    if(!computeSupervoxel(workspace))
        return false;

    init_weights("expert",2);
    return true;
}

void SurfaceOfInterest::find_soi(const PointCloudXYZ::Ptr key_pts){
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZ>());

    PointCloudT centr;
    std::map<int, uint32_t> centroids_label;
    _labels.clear();
    _labels_no_soi.clear();
    _weights["keyPts"].clear();
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
//    std::map<uint32_t,uint32_t> no_result;

    for(int i = 0; i < key_pts->size(); i++){
        if(!pcl::isFinite(key_pts->points[i]))
            continue;

        if(!tree->nearestKSearch(key_pts->points[i],1,nn_indices,nn_distance)){
            PCL_WARN ("No neighbor found for point %lu (%f %f %f)!\n",
                      i, key_pts->points[i].x, key_pts->points[i].y, key_pts->points[i].z);
            continue;
        }

        result.emplace(centroids_label[nn_indices[0]],centroids_label[nn_indices[0]]);

//        if(nn_distance[0] < 0.05)
//            result.emplace(centroids_label[nn_indices[0]],centroids_label[nn_indices[0]]);
//        else
//            no_result.emplace(centroids_label[nn_indices[0]],centroids_label[nn_indices[0]]);
    }


    std::vector<double> v = {0.,1.};
    for(auto it = result.begin(); it != result.end(); it++)
        _weights["keyPts"].emplace(it->first,v);


//    for(auto it = no_result.begin(); it != no_result.end(); it++)
//        _labels_no_soi.push_back(it->first);

//    for(auto it = result.begin(); it != result.end(); it++){
//        _labels.push_back(it->first);
//        _weights.emplace(it->first,1.);
//    }
//    assert(_labels.size() == _weights.size());
}

void SurfaceOfInterest::init_weights(const std::string& modality, int nbr_class, float value){
//    for(auto& mod : _weights){
//        mod.second.clear();
    _weights[modality].clear();
    for(auto it_sv = _supervoxels.begin(); it_sv != _supervoxels.end(); it_sv++){
        _weights[modality].emplace(it_sv->first,std::vector<double>(nbr_class,value));
    }
}

void SurfaceOfInterest::reduce_to_soi(){
    for(int i = 0; i < _labels_no_soi.size(); i++){
        remove(_labels_no_soi[i]);
    }
}

bool SurfaceOfInterest::choice_of_soi(const std::string& modality, pcl::Supervoxel<PointT> &supervoxel, uint32_t &lbl){

    //*build the distribution from weights
    std::map<float,uint32_t> soi_dist;
    float val = 0.f;
    float total_w = 0.f;
    for(auto it = _weights[modality].begin(); it != _weights[modality].end(); it++)
        total_w += it->second[lbl];

    if(total_w == 0)
        return false;


    for(auto it = _weights[modality].begin(); it != _weights[modality].end(); it++){
        val+=it->second[lbl]/(total_w/**_weights[modality].size()*/);
        soi_dist.emplace(val,it->first);
    }
    //*/


    //*choice of a supervoxel
    boost::random::uniform_real_distribution<> dist(0.,1.);

    float choice = dist(_gen);

    lbl = soi_dist.lower_bound(choice)->second;
    assert(_supervoxels[lbl]);
    supervoxel = *(_supervoxels[lbl]);
    //*/

    return true;
}

bool SurfaceOfInterest::choice_of_soi_by_uncertainty(const std::string& modality, pcl::Supervoxel<PointT> &supervoxel, uint32_t &lbl){

    //*build the distribution from weights
    std::map<float,uint32_t> soi_dist;
    float val = 0.f;

    float total_w = 0.f;
    for(auto it = _weights[modality].begin(); it != _weights[modality].end(); it++){
        if(it->second[lbl] > 0.5)
            total_w += (1.-it->second[lbl])*2.;
        else
            total_w += it->second[lbl]*2.;
    }

    std::cout << "global uncertainty : " << total_w << std::endl;

    if(total_w == 0)
        return false;

    for(auto it = _weights[modality].begin(); it != _weights[modality].end(); it++){
        if(it->second[lbl] > .5)
            val+=(1.-it->second[lbl])*2./(total_w);
        else val+=(it->second[lbl])*2./(total_w);
        soi_dist.emplace(val,it->first);
    }
    //*/

    //*choice of a supervoxel
    boost::random::uniform_real_distribution<> dist(0.,1.);

    float choice = dist(_gen);
    lbl = soi_dist.lower_bound(choice)->second;
    supervoxel = *(_supervoxels[lbl]);
    //*/

    return true;
}


void SurfaceOfInterest::delete_background(const PointCloudT::Ptr background){
    pcl::KdTreeFLANN<PointT>::Ptr tree(new pcl::KdTreeFLANN<PointT>);
    tree->setInputCloud(background);

    std::vector<int> nn_indices(1);
    std::vector<float> nn_distance(1);

    PointCloudT filtered_cloud;

    for(auto pt : _inputCloud->points){
        if(!pcl::isFinite(pt))
            continue;

        if(!tree->nearestKSearch(pt,1,nn_indices,nn_distance))
            continue;

        if(nn_distance[0] > 0.0001)
            filtered_cloud.points.push_back(pt);
    }

    _inputCloud.reset(new PointCloudT(filtered_cloud));
}

pcl::PointCloud<pcl::PointXYZI> SurfaceOfInterest::getColoredWeightedCloud(const std::string &modality,int lbl){

    pcl::PointCloud<pcl::PointXYZI> result;
    pcl::PointXYZI pt;

    for(auto it_sv = _supervoxels.begin(); it_sv != _supervoxels.end(); it_sv++){
        pcl::Supervoxel<PointT>::Ptr current_sv = it_sv->second;
        float c = _weights[modality][it_sv->first][lbl];

        for(auto v : *(current_sv->voxels_)){
            pt.x = v.x;
            pt.y = v.y;
            pt.z = v.z;
            pt.intensity = c;
            result.push_back(pt);
        }
    }

    return result;
}


std::map<pcl::Supervoxel<PointT>::Ptr, int> SurfaceOfInterest::get_supervoxels_clusters(const std::string &modality, double &saliency_threshold,int lbl){
    std::map<pcl::Supervoxel<PointT>::Ptr, int> sv_clusters;

    int cluster_id = 0;

    std::function<void (uint32_t, int)> _add_supervoxels_to_clusters = [&](uint32_t sv_label, int cluster_id) {
      double weight = _weights[modality][sv_label][lbl];
      pcl::Supervoxel<PointT>::Ptr sv = _supervoxels.find(sv_label)->second;
      auto it = sv_clusters.find(sv);

      if (weight > saliency_threshold && it == sv_clusters.end()) {
          sv_clusters[sv] = cluster_id;

          for (auto adj_it = _adjacency_map.equal_range(sv_label).first;
               adj_it != _adjacency_map.equal_range(sv_label).second; adj_it++) {
              _add_supervoxels_to_clusters(adj_it->second, cluster_id);
          }
      }

    };

    for (auto it = _supervoxels.begin(); it != _supervoxels.end(); it++){
        _add_supervoxels_to_clusters(it->first, cluster_id);

        cluster_id += 1;
    }

    return sv_clusters;
}

void SurfaceOfInterest::neighbor_bluring(const std::string& modality, double cst,int lbl){
    relevance_map_t weights = _weights[modality];
    for(auto it_sv = _supervoxels.begin(); it_sv != _supervoxels.end(); it_sv++){
        auto neighbors = _adjacency_map.equal_range(it_sv->first);
        for(auto adj_it = neighbors.first; adj_it != neighbors.second; adj_it++){
            if(_weights[modality][adj_it->second][lbl] >= 0.5)
                weights[adj_it->first][lbl] += cst;
            else
                weights[adj_it->first][lbl] -= cst;
            if(weights[adj_it->first][lbl] >= 1.)
                weights[adj_it->first][lbl] = 1.;
            else if(weights[adj_it->first][lbl] <= 0.)
                weights[adj_it->first][lbl] = 0.;
        }
    }
    _weights[modality] = weights;
}

void SurfaceOfInterest::adaptive_threshold(const std::string& modality, int lbl){
    relevance_map_t weights = _weights[modality];
    for(auto it_sv = _supervoxels.begin(); it_sv != _supervoxels.end(); it_sv++){
        auto neighbors = _adjacency_map.equal_range(it_sv->first);
        double avg = _weights[modality][it_sv->first][lbl];
        double tot = 1;
        for(auto adj_it = neighbors.first; adj_it != neighbors.second; adj_it++){
            avg+=_weights[modality][adj_it->second][lbl];
            tot+=1.;
        }
        avg = avg/tot;
        if(avg >= 0.5 && _weights[modality][it_sv->first][lbl] >= avg)
            weights[it_sv->first][lbl] = 1.;
        else weights[it_sv->first][lbl] = 0.;

    }
    _weights[modality] = weights;
}

pcl::PointCloud<pcl::PointXYZI> SurfaceOfInterest::cumulative_relevance_map(std::vector<pcl::PointCloud<pcl::PointXYZI>> list_weights){
    pcl::PointCloud<pcl::PointXYZI> output_cloud = list_weights[0];
    for(int i = 0; i < list_weights[0].size(); i++){
        double avg = 0;
        for(const pcl::PointCloud<pcl::PointXYZI>& map: list_weights){
            avg += map[i].intensity;
        }
        avg = avg/(double)list_weights.size();
        output_cloud[i].intensity = avg;
    }
    return output_cloud;
}
