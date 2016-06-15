#include <SurfaceOfInterest.h>

using namespace image_processing;

bool SurfaceOfInterest::generate(const workspace_t &workspace){
    if(!computeSupervoxel(workspace))
        return false;
    init_weights();
    return true;
}

bool SurfaceOfInterest::generate(const TrainingData<pcl::Supervoxel<PointT>>& dataset, const workspace_t& workspace){
    if(!computeSupervoxel(workspace))
        return false;
    compute_weights(dataset);
    return true;
}

bool SurfaceOfInterest::generate(const PointCloudXYZ::Ptr key_pts, const workspace_t &workspace){
    if(!computeSupervoxel(workspace))
        return false;
    init_weights(0.);
    find_soi(key_pts);
    return true;
}

bool SurfaceOfInterest::generate(const PointCloudT::Ptr background, const workspace_t &workspace){
    delete_background(background);
    if(!computeSupervoxel(workspace))
        return false;
    init_weights();
    return true;
}


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
        _weights.emplace(it->first,1.);


//    for(auto it = no_result.begin(); it != no_result.end(); it++)
//        _labels_no_soi.push_back(it->first);

//    for(auto it = result.begin(); it != result.end(); it++){
//        _labels.push_back(it->first);
//        _weights.emplace(it->first,1.);
//    }
//    assert(_labels.size() == _weights.size());
}

void SurfaceOfInterest::init_weights(float value){
    _weights.clear();
    for(auto it_sv = _supervoxels.begin(); it_sv != _supervoxels.end(); it_sv++){
        _weights.emplace(it_sv->first,value);
    }

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
    float total_w = 0.f;
    for(auto it = _weights.begin(); it != _weights.end(); it++)
        total_w += it->second;

    for(auto it = _weights.begin(); it != _weights.end(); it++){
        val+=it->second/(total_w);
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

void SurfaceOfInterest::compute_weights(const TrainingData<pcl::Supervoxel<PointT>>& data){
    init_weights();

    for(int i = 0; i < data.size(); i++){

        bool interest = data[i].first;
        pcl::Supervoxel<PointT> sv = data[i].second;


        std::map<uint32_t,float> distances;
        _compute_distances(distances,sv);

//        for(auto it_sv = _supervoxels.begin(); it_sv != _supervoxels.end(); it_sv++){
//            VectorXd sv_sample;
//            sv_sample(0) = it_sv->second->centroid_.x;
//            sv_sample(1) = it_sv->second->centroid_.y;
//            sv_sample(2) = it_sv->second->centroid_.z;
//            sv_sample(3) = it_sv->second->normal_.normal[0];
//            sv_sample(4) = it_sv->second->normal_.normal[1];
//            sv_sample(5) = it_sv->second->normal_.normal[2];
//            distances.emplace(it_sv->first,(sv_sample - sample).squaredNorm());
//        }


        for(auto it_w = _weights.begin(); it_w != _weights.end(); it_w++){

            float dist;
            if(interest)
                dist = distances[it_w->first];
            else
                dist = 1 - distances[it_w->first];

            if(dist < _param.distance_threshold)
                continue;

            it_w->second = it_w->second - _param.interest_increment*dist;
            if(it_w->second < 0)
                it_w->second = 0.;
        }


    }


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

void SurfaceOfInterest::_compute_distances(std::map<uint32_t, float> &distances, pcl::Supervoxel<PointT> ref_sv){

    std::vector<float> ref_normal = {ref_sv.normal_.normal[0],
                                     ref_sv.normal_.normal[1],
                                     ref_sv.normal_.normal[2],
                                     ref_sv.normal_.normal[3]};

    std::vector<float> ref_rgb = {(float)ref_sv.centroid_.r,
                                  (float)ref_sv.centroid_.g,
                                  (float)ref_sv.centroid_.b};

    float w = _param.color_normal_ratio;
    float v = 1-_param.color_normal_ratio;


    auto it_sv = _supervoxels.begin();

    pcl::Supervoxel< PointT >::Ptr sv = it_sv->second;
    std::vector<float> normal = {sv->normal_.normal[0],
                                 sv->normal_.normal[1],
                                 sv->normal_.normal[2],
                                 sv->normal_.normal[3]};

    std::vector<float> rgb = {(float)sv->centroid_.r,
                              (float)sv->centroid_.g,
                              (float)sv->centroid_.b};
    float color_distance = _L2_distance(ref_rgb,rgb)/255.;
    float normal_distance = _L2_distance(ref_normal,normal)/2.;
    float distance = sqrt(w*color_distance*color_distance+v*normal_distance*normal_distance);
    distances.emplace(it_sv->first,distance);
    float max_distance = distance;

    for(; it_sv != _supervoxels.end(); ++it_sv){
        //        if(it_sv->first == lbl)
        //            continue;
        sv = it_sv->second;
        std::vector<float> normal = {sv->normal_.normal[0],
                                     sv->normal_.normal[1],
                                     sv->normal_.normal[2],
                                     sv->normal_.normal[3]};


        std::vector<float> rgb = {(float)sv->centroid_.r,
                                  (float)sv->centroid_.g,
                                  (float)sv->centroid_.b};
        color_distance = _L2_distance(ref_rgb,rgb)/255.;
        normal_distance = _L2_distance(ref_normal,normal)/2.;
        distance = sqrt(w*color_distance*color_distance+v*normal_distance*normal_distance);
        if(distance > max_distance)
            max_distance = distance;
        distances.emplace(it_sv->first,distance);
    }

    for(auto it_d = distances.begin(); it_d != distances.end(); it_d++)
        it_d->second = it_d->second / max_distance;
}

float SurfaceOfInterest::_L2_distance(const std::vector<float> &p1, const std::vector<float> &p2){

    float square_sum = 0;
    for(int i = 0; i < p1.size() ; i++){
        square_sum += (p1[i]-p2[i])*(p1[i]-p2[i]);
    }
    return sqrt(square_sum);

}

PointCloudT SurfaceOfInterest::getColoredWeightedCloud(){

    PointCloudT result;

    for(auto it_sv = _supervoxels.begin(); it_sv != _supervoxels.end(); it_sv++){
        pcl::Supervoxel<PointT>::Ptr current_sv = it_sv->second;
        float c = 255.*_weights[it_sv->first];
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
