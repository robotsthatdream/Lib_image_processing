cmake_minimum_required(VERSION 2.8)
project(image_processing)

set(SOURCE_FILES
    src/MotionDetection.cpp
    src/SupervoxelSet.cpp
    src/SurfaceOfInterest.cpp
    src/BabblingDataset.cpp
    src/HistogramFactory.cpp
    src/Object.cpp
    src/tools.cpp
)

FILE(GLOB_RECURSE HEADFILES "include/*.hpp" "include/*.h")
add_custom_target(headers_image_processing SOURCES ${HEADFILES})

set(OpenCVRequiredComponents core features2d imgcodecs video)

option(WITH_CVNONFREE "Include parts requiring non-free dependencies (currently, OpenCV Sift)." FALSE)
if(WITH_CVNONFREE)
	list(APPEND OpenCVRequiredComponents nonfree)
  list(APPEND SOURCE_FILES src/DescriptorExtraction.cpp)
endif(WITH_CVNONFREE)

#Libraries
find_package(PCL 1.7 REQUIRED)
#set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.54 COMPONENTS filesystem serialization system thread iostreams chrono REQUIRED)
# OpenCV: at least 2.4? Anyway, formally requiring OpenCV 2.4 rejects 3.2 in practice.
find_package(OpenCV REQUIRED COMPONENTS ${OpenCVRequiredComponents}) 
find_package(VTK REQUIRED)

include(CMake/FindTBB.cmake)

SET(CMM_INSTALL_TREE FALSE CACHE STRING "Path to CMM.")

if(NOT CMM_INSTALL_TREE)
  message(SEND_ERROR "Please tell me where to find a compiled CMM.  For example:
      cmake -DCMM_INSTALL_TREE=/path/to/cmm.installtree/ .
Then you can run again.")
endif(NOT CMM_INSTALL_TREE)

add_library(cmm SHARED IMPORTED)
set_property(TARGET cmm PROPERTY IMPORTED_LOCATION "${CMM_INSTALL_TREE}/lib/libcmm.so" )
# The /lib/libcmm.so above could be further cleaned up.


# # find_library(CMM_LIBRARY cmm)
# find_package(cmm)

include_directories(${PCL_INCLUDE_DIRS} /usr/local/include include ${OpenCV_INCLUDE_DIRS} "${CMM_INSTALL_TREE}/include" )
link_directories(${PCL_LIBRARY_DIRS} /usr/local/lib)
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 ${CMAKE_CXX_FLAGS}")
set(LIBRARY_OUTPUT_PATH lib/${CMAKE_BUILD_TYPE})

install(DIRECTORY include/image_processing/  DESTINATION include/${PROJECT_NAME})

file(GLOB_RECURSE src_files src/* include/image_processing/*)

add_library(image_processing SHARED  ${SOURCE_FILES})
target_link_libraries(image_processing  ${PCL_LIBRARIES} yaml-cpp.so ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} tbb)

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)

#test and example code
#add_executable(load_dataset_test test/load_dataset_test.cpp)
#target_link_libraries(load_dataset_test image_processing ${PCL_LIBRARIES} yaml-cpp.so ${OpenCV_LIBRARIES})

#add_executable(test_rgb_reconstruct test/test_rgb_reconstruct.cpp)
#target_link_libraries(test_rgb_reconstruct image_processing ${PCL_LIBRARIES} yaml-cpp.so ${OpenCV_LIBRARIES})

#add_executable(fake_data test/fake_data.cpp)
#target_link_libraries(fake_data ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

#add_executable(convex_hull_test test/convex_hull_test.cpp)
#target_link_libraries(convex_hull_test  image_processing ${PCL_LIBRARIES} oml_rf yaml-cpp.so ${OpenCV_LIBRARIES})

#add_executable(rgb_to_hsv test/rgb_to_hsv.cpp)
#target_link_libraries(rgb_to_hsv  image_processing)

#add_executable(color_hist_test test/color_hist_test.cpp)
#target_link_libraries(color_hist_test  image_processing ${OpenCV_LIBRARIES})

#add_executable(color_hist_supervoxel test/color_hist_supervoxel.cpp)
#target_link_libraries(color_hist_supervoxel  image_processing ${OpenCV_LIBRARIES})

add_executable(test_object_hyp test/test_object_hyp.cpp)
target_link_libraries(test_object_hyp  image_processing cmm tbb)
