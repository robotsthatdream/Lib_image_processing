#include "image_processing/SurfaceOfInterest.h"
#include "eigen3/Eigen/Core"
#include "opencv2/ocl/ocl.hpp"

using namespace image_processing;

//NAIVE POLICY
bool SurfaceOfInterest::generate(workspace_t &workspace){
    if(!computeSupervoxel(workspace))
	return false;

    init_weights("random");
    return true;
}

//KEYPOINTS POLICY
bool SurfaceOfInterest::generate(const PointCloudXYZ::Ptr key_pts, workspace_t &workspace){
    if(!computeSupervoxel(workspace))
	return false;

    init_weights("keyPts",0.);
    find_soi(key_pts);
    return true;
}

//EXPERT POLICY
bool SurfaceOfInterest::generate(const PointCloudT::Ptr background, workspace_t &workspace){
    delete_background(background);
    if(!computeSupervoxel(workspace))
	return false;

    init_weights("expert");
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


    for(auto it = result.begin(); it != result.end(); it++)
        _weights["keyPts"].emplace(it->first,1.);


//    for(auto it = no_result.begin(); it != no_result.end(); it++)
//        _labels_no_soi.push_back(it->first);

//    for(auto it = result.begin(); it != result.end(); it++){
//        _labels.push_back(it->first);
//        _weights.emplace(it->first,1.);
//    }
//    assert(_labels.size() == _weights.size());
}

void SurfaceOfInterest::init_weights(const std::string& modality, float value){
//    for(auto& mod : _weights){
//        mod.second.clear();
    _weights[modality].clear();
    for(auto it_sv = _supervoxels.begin(); it_sv != _supervoxels.end(); it_sv++){
        _weights[modality].emplace(it_sv->first,value);
    }
}

void SurfaceOfInterest::reduce_to_soi(){
    for(int i = 0; i < _labels_no_soi.size(); i++){
        remove(_labels_no_soi[i]);
    }
    consolidate();
}

bool SurfaceOfInterest::choice_of_soi(const std::string& modality, pcl::Supervoxel<PointT> &supervoxel, uint32_t &lbl){

    //*build the distribution from weights
    std::map<float,uint32_t> soi_dist;
    float val = 0.f;
    float total_w = 0.f;
    for(auto it = _weights[modality].begin(); it != _weights[modality].end(); it++)
        total_w += it->second;

    if(total_w == 0)
        return false;


    for(auto it = _weights[modality].begin(); it != _weights[modality].end(); it++){
        val+=it->second/(total_w/**_weights[modality].size()*/);
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
        if(it->second > 0.5)
            total_w += (1.-it->second)*2.;
        else
            total_w += it->second*2.;
    }

    std::cout << "global uncertainty : " << total_w << std::endl;

    if(total_w == 0)
        return false;

    for(auto it = _weights[modality].begin(); it != _weights[modality].end(); it++){
        if(it->second > .5)
            val+=(1.-it->second)*2./(total_w);
        else val+=(it->second)*2./(total_w);
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

PointCloudT SurfaceOfInterest::getColoredWeightedCloud(const std::string &modality){

    PointCloudT result;

    for(auto it_sv = _supervoxels.begin(); it_sv != _supervoxels.end(); it_sv++){
        pcl::Supervoxel<PointT>::Ptr current_sv = it_sv->second;
        float c = 255.*_weights[modality][it_sv->first];
        uint8_t color = c;

        for(auto v : *(current_sv->voxels_)){
            PointT pt;
            pt.x = v.x;
            pt.y = v.y;
            pt.z = v.z;
            pt.r = color;
            pt.g = color;
            pt.b = color;
            result.push_back(pt);
        }
    }

    return result;
}

PointCloudT SurfaceOfInterest::getColoredWeightedCloud(saliency_map_t &map){

    PointCloudT result;

    for(const auto& e : map){
        pcl::Supervoxel<PointT>::Ptr current_sv = _supervoxels[e.first];
        float c = 255.*e.second;
        uint8_t color = c;

        for(auto v : *(current_sv->voxels_)){
            PointT pt;
            pt.x = v.x;
            pt.y = v.y;
            pt.z = v.z;
            pt.r = color;
            pt.g = color;
            pt.b = color;
            result.push_back(pt);
        }
    }

    return result;
}

std::vector<std::set<uint32_t>> SurfaceOfInterest::extract_regions(const std::string &modality, double saliency_threshold)
{
    std::set<uint32_t> labels_set;
    std::vector<std::set<uint32_t>> regions;

    std::function<void (std::set<uint32_t>&, uint32_t)> _add_supervoxels_to_region = [&](std::set<uint32_t>& region, uint32_t sv_label) {
        double weight = _weights[modality][sv_label];
        auto it = labels_set.find(sv_label);

        if (weight > saliency_threshold && it == labels_set.end()) {
            labels_set.insert(sv_label);
            region.insert(sv_label);

            for (auto adj_it = _adjacency_map.equal_range(sv_label).first;
                 adj_it != _adjacency_map.equal_range(sv_label).second; adj_it++) {
                _add_supervoxels_to_region(region, adj_it->second);
            }
        }
    };

    for (auto it = _supervoxels.begin(); it != _supervoxels.end(); it++){
        std::set<uint32_t> region;
        _add_supervoxels_to_region(region, it->first);

        if (region.size() > 0) {
            regions.push_back(region);
        }
    }

    return regions;
}

size_t SurfaceOfInterest::get_closest_region(const std::vector<std::set<uint32_t>> regions, const Eigen::Vector4d center)
{
    if (regions.size() > 0) {
        int closest_i;
        double closest_d = std::numeric_limits<double>::max();
        for (int i = 0; i < regions.size(); i++) {
            Eigen::Vector4d r_center;
            pcl::compute3DCentroid<PointT>(get_cloud(regions[i]), r_center);
            double dx = r_center[0] - center[0];
            double dy = r_center[1] - center[1];
            double dz = r_center[2] - center[2];
            double d = dx*dx + dy*dy + dz*dz;
            if (d < closest_d) {
                closest_i = i;
                closest_d = d;
            }
        }

        return closest_i;
    }
    else {
        return -1;
    }
}

std::set<uint32_t> SurfaceOfInterest::extract_background(const std::string &modality, double saliency_threshold)
{
    std::set<uint32_t> background;

    for (auto it = _supervoxels.begin(); it != _supervoxels.end(); it++){
        double weight = _weights[modality][it->first];
        if (weight < saliency_threshold) {
            background.insert(it->first);
        }
    }

    return background;
}
