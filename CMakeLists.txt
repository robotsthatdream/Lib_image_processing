cmake_minimum_required(VERSION 2.8)
project(image_processing)

set(SOURCE_FILES src/DescriptorExtraction.cpp
    src/MotionDetection.cpp
    src/SupervoxelSet.cpp
    src/SurfaceOfInterest.cpp
    src/BabblingDataset.cpp
    src/tools.cpp
)

#Libraries
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS filesystem system REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS features2d nonfree)
find_package(PCL 1.7 REQUIRED)
find_package(VTK REQUIRED)

include_directories(include ${PCL_INCLUDE_DIRS} /usr/local/include)
link_directories(${PCL_LIBRARY_DIRS} /usr/local/lib)
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(LIBRARY_OUTPUT_PATH lib/${CMAKE_BUILD_TYPE})

install(DIRECTORY include/image_processing/  DESTINATION /usr/local/include/${PROJECT_NAME})

file(GLOB_RECURSE src_files src/* include/image_processing/*)

add_library(image_processing ${SOURCE_FILES})
target_link_libraries(image_processing  ${PCL_LIBRARIES} oml_rf yaml-cpp.so ${Boost_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION /usr/local/lib
    LIBRARY DESTINATION /usr/local/lib)

#test and example code
add_executable(load_dataset_test test/load_dataset_test.cpp)
target_link_libraries(load_dataset_test image_processing ${PCL_LIBRARIES} oml_rf yaml-cpp.so ${OpenCV_LIBRARIES})

add_executable(test_rgb_reconstruct test/test_rgb_reconstruct.cpp)
target_link_libraries(test_rgb_reconstruct image_processing ${PCL_LIBRARIES} oml_rf yaml-cpp.so ${OpenCV_LIBRARIES})

add_executable(fake_data test/fake_data.cpp)
target_link_libraries(fake_data ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(convex_hull_test test/convex_hull_test.cpp)
target_link_libraries(convex_hull_test  image_processing ${PCL_LIBRARIES} oml_rf yaml-cpp.so ${OpenCV_LIBRARIES})

add_executable(rgb_to_hsv test/rgb_to_hsv.cpp)
target_link_libraries(rgb_to_hsv  image_processing)
