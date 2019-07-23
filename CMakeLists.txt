cmake_minimum_required(VERSION 2.8)
project(image_processing)

#================================================================
# Reproducibility: make sure what we compile is a known git commit.
# It will be logged.
#================================================================
set (GIT_EXECUTABLE "git")

# This ensures that what is compiled is something committed and identifiable with a git hash.
#
# This includes https://stackoverflow.com/questions/38823537/execute-git-describe-in-custom-target
# Store version into variable
execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  COMMAND ${GIT_EXECUTABLE} diff-index HEAD RESULT_VARIABLE ret)
if(NOT ret EQUAL "0")
  message( "This is fine.")
#    message( FATAL_ERROR "Build tree not clean.  Please commit any needed change and remove the rest.")
endif()

execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  COMMAND ${GIT_EXECUTABLE} describe --tags --always
  OUTPUT_VARIABLE GIT_REPO_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message( "GIT_REPO_VERSION=${GIT_REPO_VERSION}")


set (GIT_VERSION_STRING_INCLUDE_DIRECTORY "${PROJECT_BINARY_DIR}/git_version")
file(MAKE_DIRECTORY ${GIT_VERSION_STRING_INCLUDE_DIRECTORY})
configure_file("git_version.hpp.in" "${GIT_VERSION_STRING_INCLUDE_DIRECTORY}/git_version.hpp" ESCAPE_QUOTES)
#================================================================


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

set(OpenCVRequiredComponents core features2d video)

option(WITH_CVNONFREE "Include parts requiring non-free dependencies (currently, OpenCV Sift)." FALSE)
if(WITH_CVNONFREE)
	list(APPEND OpenCVRequiredComponents nonfree)
  list(APPEND SOURCE_FILES src/DescriptorExtraction.cpp)
endif(WITH_CVNONFREE)

#Libraries
#set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.54 COMPONENTS filesystem system thread iostreams chrono REQUIRED)
# OpenCV: at least 2.4? Anyway, formally requiring OpenCV 2.4 rejects 3.2 in practice.
find_package(OpenCV REQUIRED COMPONENTS ${OpenCVRequiredComponents})
find_package(PCL 1.7 REQUIRED)
find_package(VTK REQUIRED)

include(CMake/FindTBB.cmake)

SET(IAGMM_INSTALL_TREE FALSE CACHE STRING "Path to IAGMM.")

if(NOT IAGMM_INSTALL_TREE)
  message(SEND_ERROR "Please tell me where to find a compiled IAGMM.  For example:
      cmake -DIAGMM_INSTALL_TREE=/path/to/iagmm.installtree/ .
Then you can run again.")
endif(NOT IAGMM_INSTALL_TREE)

add_library(iagmm SHARED IMPORTED)
set_property(TARGET iagmm PROPERTY IMPORTED_LOCATION "${IAGMM_INSTALL_TREE}/lib/libiagmm.so" )
# The /lib/libiagmm.so above could be further cleaned up.


# # find_library(IAGMM_LIBRARY iagmm)
# find_package(iagmm)

include(CMake/FindLibCmaes.cmake)

include_directories(${PCL_INCLUDE_DIRS} /usr/local/include include ${OpenCV_INCLUDE_DIRS} "${IAGMM_INSTALL_TREE}/include" "${LIBCMAES_INCLUDE_DIR}" ${GIT_VERSION_STRING_INCLUDE_DIRECTORY})
link_directories(${PCL_LIBRARY_DIRS} /usr/local/lib)
add_definitions(${PCL_DEFINITIONS})

add_compile_options("$<$<CONFIG:RELEASE>:-O3>")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options("-W")
  add_compile_options("-Wall")
  add_compile_options("-Wextra")
  add_compile_options("-Wno-vla")
  add_compile_options("-Wno-reorder")
#  add_compile_options("-Werror")
  add_compile_options("-Wno-unused-parameter")
endif ()

if (CMAKE_VERSION VERSION_LESS "3.1")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options("-std=c++11")
  endif ()
else ()
  set (CMAKE_CXX_STANDARD 11)
endif ()

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

message(STATUS "LIBCMAES_LIBRARIES=${LIBCMAES_LIBRARIES}")
message(STATUS "LIBCMAES_LIBRARIES=${LIBCMAES_LIBRARIES}")

add_executable(test_object_hyp test/test_object_hyp.cpp test/test_rotation.cpp)
target_link_libraries(test_object_hyp  image_processing iagmm ${IAGMM_Lib_LIBRARIES} tbb ${LIBCMAES_LIBRARIES})
#target_include_directories(test_object_hyp ${LIBCMAES_INCLUDE_DIR})




# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()
# CMake in Ubuntu 18.10 fails without this line: include_directories("${gtest_SOURCE_DIR}/include")

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(test_rotation test/test_rotation.cpp)
target_compile_definitions(test_rotation PRIVATE -DFSG_COMPILING_GTEST_EXECUTABLE=1)
target_link_libraries(test_rotation gtest_main)
add_test(NAME test_rotation COMMAND test_rotation)



################################################################
# Log facility
################################################################
# Allow to have relative source names in text logs.
# VERY comfortable to jump from log to source code in a keypress.
# This currently uses emacs conventions, might work in other environments.

#message(STATUS "will strip ${COUNT_OF_CHARS_TO_STRIP_TO_GET_PATH_RELATIVE_TO_CMAKE_HOME_DIRECTORY}")
foreach(SOURCE_FILE IN ITEMS test/test_object_hyp.cpp test/test_rotation.cpp)
  set_source_files_properties(${SOURCE_FILE} PROPERTIES
    COMPILE_DEFINITIONS "FSG_PROJECT_ROOT=\"${CMAKE_HOME_DIRECTORY}\";FSG_PROJECT_RELATIVE_PATHNAME=\"${SOURCE_FILE}\"")
  #message(STATUS "${RELPATHNAME}")
endforeach()
################################################################

# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
