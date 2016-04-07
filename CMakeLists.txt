cmake_minimum_required(VERSION 2.8)
project(image_processing)

set(SOURCE_FILES src/DescriptorExtraction.cpp src/MotionDetection.cpp src/SupervoxelSet.cpp src/SurfaceOfInterest.cpp)

#Libraries
find_package(Boost REQUIRED COMPONENTS system )
find_package(OpenCV REQUIRED COMPONENTS features2d nonfree)
find_package(PCL 1.7 REQUIRED)
find_package(VTK REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(LIBRARY_OUTPUT_PATH lib/${CMAKE_BUILD_TYPE})

include_directories(includes)

install(DIRECTORY includes/  DESTINATION /usr/local/include/${PROJECT_NAME})

file(GLOB_RECURSE src_files src/* includes/*)

add_library(image_processing ${SOURCE_FILES})
target_link_libraries(image_processing ${PCL_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION /usr/local/lib
    LIBRARY DESTINATION /usr/local/lib)
