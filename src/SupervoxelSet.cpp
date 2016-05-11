#include <SupervoxelSet.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/correspondence_estimation.h>
#include <boost/random.hpp>
#include <pcl/filters/passthrough.h>


using namespace image_processing;

void SupervoxelSet::computeSupervoxel(std::vector<float> area){

    if(!area.empty()){//if an area is given reduce the input cloud to the area.
        pcl::PassThrough<PointT> passFilter;


        passFilter.setInputCloud(_inputCloud);
        passFilter.setFilterFieldName("x");
        passFilter.setFilterLimits(area[0],area[1]);
        passFilter.filter(*_inputCloud);

        passFilter.setInputCloud(_inputCloud);
        passFilter.setFilterFieldName("y");
        passFilter.setFilterLimits(area[2],area[3]);
        passFilter.filter(*_inputCloud);

        passFilter.setInputCloud(_inputCloud);
        passFilter.setFilterFieldName("z");
        passFilter.setFilterLimits(area[4],area[5]);
        passFilter.filter(*_inputCloud);


    }

    //input cloud
    if(_inputCloud->empty()){
        std::cerr << "error : input cloud is empty" << std::endl;
        exit(1);
    }


    //definition of super voxel clustering class
    _extractor->setInputCloud(_inputCloud);

    //--

    std::cout << "Extracting supervoxels!" << std::endl;

    _extractor->extract(_supervoxels);
    assert(_supervoxels.size() != 0);
    _extractor->getSupervoxelAdjacency(_adjacency_map);
    std::cout << "Found " << _supervoxels.size() << " supervoxels" << std::endl;
}

void SupervoxelSet::extractEdges(PointCloudT::Ptr edges_cloud, AdjacencyMap supervoxel_adjacency){

    if(supervoxel_adjacency.empty())
        supervoxel_adjacency = _adjacency_map;

    //compute one edge point per adjacent supervoxel.
    std::multimap<uint32_t,uint32_t> verification_map;

    std::multimap<uint32_t,uint32_t>::iterator label_itr = supervoxel_adjacency.begin();
    for(;label_itr != supervoxel_adjacency.end();){//iterate through adjacency map
        uint32_t supervoxel_label = label_itr->first;
        pcl::Supervoxel<PointT>::Ptr supervoxel = _supervoxels.at(supervoxel_label);
        std::multimap<uint32_t,uint32_t>::iterator adjacent_itr = supervoxel_adjacency.equal_range(supervoxel_label).first;
        int i = 0;
        for(;adjacent_itr != supervoxel_adjacency.equal_range(supervoxel_label).second;++adjacent_itr){
            //iterate through neighborhood of current supervoxel
            pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel = _supervoxels.at (adjacent_itr->second);
            PointCloudT::Ptr neighbor_cloud = neighbor_supervoxel->voxels_;
            PointCloudT::iterator cloud_itr = neighbor_cloud->begin();
            PointT currentPoint = *cloud_itr;
            for(;cloud_itr != neighbor_cloud->end();++cloud_itr){
                PointT point = *cloud_itr;
                if(pcl::euclideanDistance(point,supervoxel->centroid_)
                        < pcl::euclideanDistance(currentPoint,supervoxel->centroid_))
                    currentPoint = point;
            }
            //to do a comment
            bool can_append = true;
            if(verification_map.find(adjacent_itr->first) != verification_map.end()){
                std::multimap<uint32_t,uint32_t>::iterator verif_itr
                        = verification_map.equal_range(adjacent_itr->first).first;
                for(;verif_itr != verification_map.equal_range(adjacent_itr->first).second;verif_itr++){
                    if(verif_itr->second == supervoxel_label)
                        can_append = false;
                }
            }
            if(can_append){
                edges_cloud->push_back(currentPoint);
                verification_map.insert(std::pair<uint32_t,uint32_t>(supervoxel_label,adjacent_itr->first));
            }
        }
        i++;
        if(i<8){//if this supervoxel is a side supervoxel
            PointCloudT::iterator cloud_itr = supervoxel->voxels_->begin();
            PointT selected_pt = *cloud_itr;
            float previous_sum =  pcl::euclideanDistance(edges_cloud->back(),selected_pt);
            for(int j = i; j > 1 ; j--){
                previous_sum+=pcl::euclideanDistance(selected_pt,edges_cloud->at(edges_cloud->size()-j));
            }
            previous_sum+=pcl::euclideanDistance(selected_pt,supervoxel->centroid_);
            for(;cloud_itr != supervoxel->voxels_->end();++cloud_itr){
                PointT point = *cloud_itr;
                float sum = pcl::euclideanDistance(edges_cloud->back(),point);
                for(int j = i; j > 1 ; j--){
                    sum+=pcl::euclideanDistance(point,edges_cloud->at(edges_cloud->size()-j));
                }
                sum += pcl::euclideanDistance(point,supervoxel->centroid_);
                if(sum > previous_sum){
                    previous_sum = sum;
                    selected_pt = point;
                }
            }
            edges_cloud->push_back(selected_pt);
        }
        label_itr = supervoxel_adjacency.upper_bound (supervoxel_label);
    }
}


void SupervoxelSet::consolidate(){
    std::cout << "consolidate" << std::endl;

    for(SupervoxelArray::iterator super_it = _supervoxels.begin();
        super_it != _supervoxels.end();super_it++){
        AdjacencyMap::iterator neighbor_it = _adjacency_map.equal_range(super_it->first).first;
        for(;neighbor_it != _adjacency_map.equal_range(super_it->first).second; neighbor_it++){
            if(!contain(neighbor_it->second))
                _adjacency_map.erase(neighbor_it->second);
        }
        //        super_it = adjacency_map.upper_bound (adja_it->first);
    }

}

bool SupervoxelSet::contain(uint32_t label){
    return _supervoxels.find(label) != _supervoxels.end();
}

uint32_t SupervoxelSet::whichVoxelContain(float x, float y, float z){
  boost::random::mt19937 gen;

  return isInThisVoxel(x,y,z,_supervoxels.begin()->first,_adjacency_map,gen,20);
}

uint32_t SupervoxelSet::isInThisVoxel(float x, float y, float z, uint32_t label,AdjacencyMap am,  boost::random::mt19937 gen, int counter ){
  if(counter <= 0)
    return 221133;
  label = _supervoxels.lower_bound(label)->first;
  pcl::Supervoxel<PointT>::Ptr vox = _supervoxels.at(label);
  double distance = sqrt((x-vox->centroid_.x)*(x-vox->centroid_.x)
                         + (y-vox->centroid_.y)*(y-vox->centroid_.y)
                         + (z-vox->centroid_.z)*(z-vox->centroid_.z));
  if(distance < _extractor->getSeedResolution())
    return label;
  else{
      AdjacencyMap::iterator it = am.equal_range(label).first;
      AdjacencyMap::iterator neighbor_it = am.equal_range(label).first;
      vox = _supervoxels.at(neighbor_it->second);
      uint32_t new_lbl = neighbor_it->second;
      double new_dist = sqrt((x-vox->centroid_.x)*(x-vox->centroid_.x)
                     + (y-vox->centroid_.y)*(y-vox->centroid_.y)
                     + (z-vox->centroid_.z)*(z-vox->centroid_.z));
      for(;neighbor_it != am.equal_range(label).second; ++neighbor_it){

          vox = _supervoxels.at(neighbor_it->second);
          double tmp = sqrt((x-vox->centroid_.x)*(x-vox->centroid_.x)
                         + (y-vox->centroid_.y)*(y-vox->centroid_.y)
                         + (z-vox->centroid_.z)*(z-vox->centroid_.z));
          if(tmp < new_dist){
            new_dist = tmp;
            new_lbl = neighbor_it->second;
            it = neighbor_it;
            }
        }
      if(new_dist > distance ){
          boost::random::uniform_int_distribution<> dist(1,_supervoxels.size());
          return isInThisVoxel(x,y,z,dist(gen),am,gen,counter-1);
        }
      //supression de l'arc parcourue
//      neighbor_it = am.equal_range(new_lbl).first;
//      for(;neighbor_it != am.equal_range(new_lbl).second; ++neighbor_it){
//          if(neighbor_it->second == label)
//            break;
//        }
//      am.erase(neighbor_it);
//      am.erase(it);
      return isInThisVoxel(x,y,z,new_lbl,am,gen,counter);
    }
}

void SupervoxelSet::insert(uint32_t label,
                               pcl::Supervoxel<PointT>::Ptr supervoxel,
                               std::vector<uint32_t> neighborLabel){
    _supervoxels.insert(std::pair<uint32_t,pcl::Supervoxel<PointT>::Ptr >(label,supervoxel));

    for(int i = 0; i < neighborLabel.size(); i++)
        _adjacency_map.insert(std::pair<uint32_t,uint32_t>(label,neighborLabel.at(i)));
}


void SupervoxelSet::remove(uint32_t label){
    _supervoxels.erase(label);
    _adjacency_map.erase(label);
}

void SupervoxelSet::clear(){
    for(auto it = _supervoxels.begin();it != _supervoxels.end();it++){
        remove(it->first);
    }
    _extractor.reset(new pcl::SupervoxelClustering<PointT>(supervoxel::voxel_resolution,supervoxel::seed_resolution,supervoxel::use_transform));
}

void SupervoxelSet::extractCloud(PointCloudT& resultCloud){
    for(SupervoxelArray::iterator sv_itr = _supervoxels.begin();
        sv_itr != _supervoxels.end(); sv_itr++){
        pcl::Supervoxel<PointT>::Ptr current = sv_itr->second;
        resultCloud += *(current->voxels_);
    }


}

SupervoxelSet SupervoxelSet::compare(SupervoxelSet &super,double threshold,double position_importance, double normal_importance){
    pcl::KdTreeFLANN<PointT>::Ptr tree(new pcl::KdTreeFLANN<PointT>());
//    pcl::KdTreeFLANN<pcl::Normal>::Ptr normalTree(new pcl::KdTreeFLANN<pcl::Normal>());

    PointCloudT::Ptr targetCloud(new PointCloudT);
    PointCloudN::Ptr targetNormalCloud(new PointCloudN);
    std::map<int,uint32_t> targetLabel;
    getCentroidCloud(*targetCloud,targetLabel,*targetNormalCloud);

    tree->setInputCloud(targetCloud);
//    normalTree->setInputCloud(targetNormalCloud);

    PointCloudT inputCentroids;
    PointCloudN inputNormals;
    std::map<int, uint32_t> inputLabel;
    super.getCentroidCloud(inputCentroids,inputLabel,inputNormals);

    std::vector<int> nn_indices(1);
    std::vector<float> nn_distance(1);

//    std::vector<int> nn_indices_normal(1);
//    std::vector<float> nn_distance_normal(1);

    SupervoxelSet resultSupervoxelSet;

    std::cout << "start search for difference" << std::endl;

    for(int i = 0; i < inputCentroids.size(); i++){
        if(!pcl::isFinite(inputCentroids.points[i]) || !pcl::isFinite(inputNormals.points[i]))
            continue;

        if(!tree->nearestKSearch(inputCentroids.points[i],1,nn_indices,nn_distance)){
            PCL_WARN ("No neighbor found for point %lu (%f %f %f)!\n",
                      i, inputCentroids.points[i].x, inputCentroids.points[i].y, inputCentroids.points[i].z);
            continue;
        }

//        if(!normalTree->nearestKSearch(inputNormals.points[i],1,nn_indices_normal,nn_distance_normal)){
//            //            PCL_WARN ("No neighbor found for point %lu (%f %f %f)!\n",
//            //                      i, inputNormals.points[i].x, inputNormals.points[i].y, inputNormals.points[i].z);
//            continue;
//        }

        if(nn_distance[0] > threshold){
//                sqrt(position_importance*nn_distance[0]*nn_distance[0] + normal_importance*nn_distance_normal[0]*nn_distance_normal[0]) > threshold){
            uint32_t currentLabel = inputLabel.at(i);
            resultSupervoxelSet.insert(currentLabel,
                                           super.at(currentLabel),
                                           super.getNeighbor(currentLabel));
        }
    }
    std::cout << "end of search" << std::endl;

    return resultSupervoxelSet;

}

Superpixels SupervoxelSet::to_superpixels(){
    Superpixels result;

    for(auto it_sv = _supervoxels.begin(); it_sv != _supervoxels.end(); it_sv++){
        std::vector<cv::Point2f> pts;
        pcl::Supervoxel<PointT> current_sv = *(it_sv->second);
        for(int i = 0; i < current_sv.voxels_->size(); i++){
            PointT pt_vx = current_sv.voxels_->at(i);
            float px = pt_vx.x;
            float py = pt_vx.y;
            float pz = pt_vx.z;
            float x = camera::focal_length_x*pt_vx.x/pt_vx.z
                    + camera::rgb_princ_pt_x;
            float y = camera::focal_length_y*pt_vx.y/pt_vx.z
                    + camera::rgb_princ_pt_y;

            pts.push_back(cv::Point2f(x,y));
        }
        result.emplace(it_sv->first,pts);
    }

    return result;
}

PointCloudT SupervoxelSet::mean_color_cloud(){
    PointCloudT ptcl;

    for(auto it_sv = _supervoxels.begin(); it_sv != _supervoxels.end(); it_sv++){
        PointCloudT::Ptr current_sv = it_sv->second->voxels_;
        PointT current_centroid = it_sv->second->centroid_;

        for(auto pt : *current_sv){
            PointT tmp_pt;
            tmp_pt.x = pt.x;
            tmp_pt.y = pt.y;
            tmp_pt.z = pt.z;
            tmp_pt.r = current_centroid.r;
            tmp_pt.g = current_centroid.g;
            tmp_pt.b = current_centroid.b;
            ptcl.push_back(pt);
        }
    }
    return ptcl;
}

void SupervoxelSet::supervoxel_to_mask(uint32_t lbl, cv::Mat &mask){

    //TODO replace hardcode values by smart values.

    PointCloudT::Ptr sv = _supervoxels.at(lbl)->voxels_;
    mask = cv::Mat::zeros(480,640,CV_8U);
    for(int i = 0; i < sv->size(); i++){
        PointT pt = sv->at(i);

        int p_x = camera::focal_length_x*pt.x/pt.z
                + camera::rgb_princ_pt_x;
        int p_y = camera::focal_length_y*pt.y/pt.z
                + camera::rgb_princ_pt_y;

        for(int k = -2; k <= 2;k++)
            for(int j = -2; j <= 2; j++)
                mask.row(p_y+k).col(p_x+j) = 255;
   }

}

void SupervoxelSet::substract(SupervoxelSet &cloud){
    SupervoxelArray cloud_sva = cloud.getSupervoxels();
    for(SupervoxelArray::iterator sv_itr = cloud_sva.begin();
        sv_itr != cloud_sva.end(); sv_itr++){
        remove(sv_itr->first);
    }


}


pcl::PointXYZ SupervoxelSet::globalPosition(){
    double sumX = 0;
    double sumY = 0;
    double sumZ = 0;
    pcl::Supervoxel<PointT>::Ptr current_sv(new pcl::Supervoxel<PointT>);
    for(SupervoxelArray::iterator itr = _supervoxels.begin();
        itr != _supervoxels.end(); itr++){
        current_sv = itr->second;
        sumX = sumX + current_sv->centroid_.x;
        sumY = sumY + current_sv->centroid_.y;
        sumZ = sumZ + current_sv->centroid_.z;
    }
    sumX = sumX/((double)_supervoxels.size());
    sumY = sumY/((double)_supervoxels.size());
    sumZ = sumZ/((double)_supervoxels.size());

    pcl::PointXYZ result(sumX,sumY,sumZ);

    return result;
}


void SupervoxelSet::getCentroidCloud(PointCloudT &centroids, std::map<int,uint32_t> &centroidsLabel, PointCloudN &centroid_normals){

    int i = 0;
    for(SupervoxelArray::iterator it = _supervoxels.begin()
        ; it != _supervoxels.end(); it++){
        pcl::Supervoxel<PointT>::Ptr supervoxel = it->second;
        centroids.push_back(supervoxel->centroid_);
        centroid_normals.push_back(supervoxel->normal_);
        centroidsLabel.insert(std::pair<int,uint32_t>(i,it->first));
        i++;
    }
}

void SupervoxelSet::getCentroidCloud(PointCloudT &centroids, std::map<int,uint32_t> &centroidsLabel){

    int i = 0;
    for(SupervoxelArray::iterator it = _supervoxels.begin()
        ; it != _supervoxels.end(); it++){
        pcl::Supervoxel<PointT>::Ptr supervoxel = it->second;
        centroids.push_back(supervoxel->centroid_);
        centroidsLabel.insert(std::pair<int,uint32_t>(i,it->first));
        i++;
    }
}

std::vector<uint32_t> SupervoxelSet::getNeighbor(uint32_t label){

    std::vector<uint32_t> result;

    AdjacencyMap::iterator neighbor_it = _adjacency_map.equal_range(label).first;
    for(;neighbor_it != _adjacency_map.equal_range(label).second;++neighbor_it)
        result.push_back(neighbor_it->second);


    return result;

}


