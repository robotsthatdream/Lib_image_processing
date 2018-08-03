#include "../include/image_processing/SurfaceOfInterest.h"
#include <boost/archive/text_iarchive.hpp>
#include <iagmm/gmm.hpp>
#include <iostream>

namespace ip = image_processing;

int main(int argc, char **argv) {

    if (argc != 3) {
        std::cerr << "Usage : \n\t- pcd file\n\t- gmm archive" << std::endl;
        return 1;
    }

    std::string pcd_file = argv[1];
    std::string gmm_archive = argv[2];

    //* Load pcd file into a pointcloud
    ip::PointCloudT::Ptr input_cloud(new ip::PointCloudT);
    pcl::io::loadPCDFile(pcd_file, *input_cloud);
    //*/

    std::cout << "pcd file loaded" << std::endl;

    //* Load the CMMs classifier from the archive
    std::ifstream ifs(gmm_archive);
    if (!ifs) {
        std::cerr << "Unable to open archive : " << gmm_archive << std::endl;
        return 1;
    }
    iagmm::GMM gmm;
    boost::archive::text_iarchive iarch(ifs);
    iarch >> gmm;
    //*/

    std::cout << "classifier archive loaded" << std::endl;

    //* Generate relevance map on the pointcloud
    ip::SurfaceOfInterest soi(input_cloud);
    std::cout << "computing supervoxel" << std::endl;
    soi.computeSupervoxel();
    std::cout << "computed supervoxel" << std::endl;
    std::cout << "computing meanFPFHLabHist" << std::endl;
    soi.compute_feature("meanFPFHLabHist");
    std::cout << "computed meanFPFHLabHist" << std::endl;
    std::cout << "computing meanFPFHLabHist weights" << std::endl;
    soi.compute_weights<iagmm::GMM>("meanFPFHLabHist", gmm);
    std::cout << "computed meanFPFHLabHist weights" << std::endl;
    //*/

    std::cout << "relevance_map extracted" << std::endl;

    //* Generate objects hypothesis
    std::vector<std::set<uint32_t>> obj_indexes;
    obj_indexes = soi.extract_regions("meanFPFHLabHist", 0.5, 1);
    //*/

    std::cout << obj_indexes.size() << " objects hypothesis extracted"
              << std::endl;

    return 0;
}
