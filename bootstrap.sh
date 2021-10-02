#!/bin/bash

set -eu

## Sanity check dependencies

TOOLS="cmake:cmake
/usr/include/eigen3:libeigen3-dev
git:git
/usr/include/boost/version.hpp:libboost-all-dev
/usr/include/vtk*:libvtk6-dev
/usr/include/tbb/tbb.h:libtbb-dev
/usr/include/yaml-cpp/yaml.h:libyaml-cpp-dev
/usr/include/qhull/qhull.h:libqhull-dev
/usr/include/lz4.h:liblz4-dev
"

# On Ubuntu 16.04, libproj-dev is an implicit dependency of vtk*.
# FIXME this has correct behavior when vtk already installed.
# As a result you may have to run this script twice.
if
    find /usr/lib/ -iname "libvtk*geovis*.so" | xargs --no-run-if-empty ldd | grep -q libproj
then
    TOOLS="$TOOLS
/usr/include/proj_api.h:libproj-dev"
fi

# Don't quote $FNAME like this "$FNAME" because it is sometimes a wildcard pattern.

MISSING=""
for TOOLNP in ${TOOLS}
do IFS=: read FNAME PNAME <<< "$TOOLNP"
   echo -ne "Checking for $FNAME \011of $PNAME...  \0011"
   if which $FNAME
   then
       true
   elif
       find $FNAME -maxdepth 0
   then
       true
   else
       MISSING="$MISSING $TOOLNP"
       echo "not found"
   fi
done
if [[ -n "$MISSING" ]]
then echo
     echo
     echo "################################"
     echo "Some tools are missing: "
     echo "$MISSING " | sed 's/:[^:]* / /g'
     echo
     echo "Suggested action(s):"
     echo
     echo -n "sudo apt-get install -y --no-install-recommends "
     echo " $MISSING " | sed 's/ [^:]*:/ /g'
     echo
     echo "Or the equivalent for your environment (yum, cygwin, etc)."
     echo
     echo "Or hack this script and report..."
     echo "################################"
     exit 1
else
    echo "######## All tools found. ########"
fi


## Set up directory hierarchy

# This prevents cmake_project_bootstrap.sh to build:
export NO_BUILD=indeed

IMAGE_PROCESSING_SOURCE_ROOT="$( cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" ; pwd ; )"

cd "${IMAGE_PROCESSING_SOURCE_ROOT}"

if [[ "$IMAGE_PROCESSING_SOURCE_ROOT" =~ \  ]] ; then echo >&2 "Refusing to build because current directory path has a space and this break some build steps: '$PWD'" ; exit 1 ; fi

chmod -c a+x ninja_with_args.sh make_with_automatic_parallel_jobs_adjustment.sh


# WARNING : this will break if build dir has a space.  The easy fix I tried did not work. -- FSG
export MY_CMAKE_GENERATOR_OPTIONS_NINJA="-G Ninja -D CMAKE_MAKE_PROGRAM:STRING=$IMAGE_PROCESSING_SOURCE_ROOT/ninja_with_args.sh "
export MY_CMAKE_GENERATOR_OPTIONS_MAKE=" -D CMAKE_MAKE_PROGRAM:STRING=$IMAGE_PROCESSING_SOURCE_ROOT/make_with_automatic_parallel_jobs_adjustment.sh "
# -G 'Unix Makefiles'

export MY_CMAKE_GENERATOR_OPTIONS="$MY_CMAKE_GENERATOR_OPTIONS_MAKE"


IMAGE_PROCESSING_BUILD_ROOT=${PWD}.dependencies_and_generated

mkdir -p "$IMAGE_PROCESSING_BUILD_ROOT"
cd "$IMAGE_PROCESSING_BUILD_ROOT"

# function append_to_variable_with_separator()
# {
# }

function compute_OS_ID()
{
    # Figure out an operating system suffix.
    export OS_ID=$( { echo $(sed -n -e 's/^ID=\(.*\)/\1/p' </etc/os-release)-$(sed -n -e 's/^VERSION_ID="\(.*\)"/\1/p' </etc/os-release) ; } | sed -e 's/[^-.a-zA-Z0-9]/_/g' ; )
    if [[ "$OS_ID" == "-" ]]
    then
        echo >&2 "WARNING: could not figure out your OS version from /etc/os-release."
        echo >&2 "WARNING: will use a generic output directory"
        OS_ID="unknown"
    fi
}

compute_OS_ID

OPENCV_IT=${IMAGE_PROCESSING_BUILD_ROOT}/opencv.OSID_${OS_ID}.installtree.Release
if [[ -d "${OPENCV_IT}" ]]
then
    echo "OpenCV already in $OPENCV_IT"
else
    (
        if [[ ! -d opencv ]]
        then
            git clone --branch 3.4.3 https://github.com/opencv/opencv/
        fi

        cd opencv
        export EXPECTED_KILOBYTES_OCCUPATION_PER_CORE=500000
        "$IMAGE_PROCESSING_SOURCE_ROOT"/cmake_project_bootstrap.sh . ${MY_CMAKE_GENERATOR_OPTIONS:-} \
                                   -D CMAKE_BUILD_TYPE:STRING=Release \
                                   -D BUILD_JAVA:BOOL=OFF \
                                   -D BUILD_PACKAGE:BOOL=OFF \
                                   -D BUILD_PERF_TESTS:BOOL=OFF \
                                   -D BUILD_PROTOBUF:BOOL=OFF \
                                   -D BUILD_opencv_apps:BOOL=OFF \
                                   -D BUILD_opencv_calib3d:BOOL=OFF \
                                   -D BUILD_opencv_dnn:BOOL=OFF \
                                   -D BUILD_opencv_java:BOOL=OFF \
                                   -D BUILD_opencv_java_bindings_generator:BOOL=OFF \
                                   -D BUILD_opencv_ml:BOOL=OFF \
                                   -D BUILD_opencv_python2:BOOL=OFF \
                                   -D BUILD_opencv_python_bindings_generator:BOOL=OFF \
                                   -D CMAKE_SKIP_INSTALL_RPATH:BOOL=OFF \
                                   -D CMAKE_SKIP_RPATH:BOOL=OFF \
                                   -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
                                   -D ENABLE_CXX11:BOOL=ON \
                                   -D ENABLE_FAST_MATH:BOOL=ON \
                                   -D WITH_1394:BOOL=OFF \
                                   -D WITH_PROTOBUF:BOOL=OFF
#        FIXME protobuf
        cd ${IMAGE_PROCESSING_BUILD_ROOT}/opencv.OSID_${OS_ID}.buildtree.Release
        time cmake --build . -- install
    )
fi
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}${CMAKE_PREFIX_PATH:+:}${OPENCV_IT}"
# echo "CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"

FLANN_IT=${IMAGE_PROCESSING_BUILD_ROOT}/flann.OSID_${OS_ID}.installtree.Release
if [[ -d "${FLANN_IT}" ]]
then
    echo "FLANN already in $FLANN_IT"
else
    echo "We don't use Ubuntu's flann library because on Ubuntu 20.04+ it causes this bug when building PCL: https://github.com/PointCloudLibrary/pcl/issues/804"
    (
        if [[ ! -d flann ]]
        then
            git clone -b 1.8.4 https://github.com/flann-lib/flann

            # Workaround https://github.com/flann-lib/flann/issues/369
            (
                cd flann
                touch src/cpp/empty.cpp
                sed -e '/add_library(flann_cpp SHARED/ s/""/empty.cpp/' \
                    -e '/add_library(flann SHARED/ s/""/empty.cpp/' \
                    -i src/cpp/CMakeLists.txt
            )
        fi

        cd flann
        export EXPECTED_KILOBYTES_OCCUPATION_PER_CORE=1900000
        "$IMAGE_PROCESSING_SOURCE_ROOT"/cmake_project_bootstrap.sh . ${MY_CMAKE_GENERATOR_OPTIONS:-} \
                                       -DCMAKE_BUILD_TYPE:STRING=Release \
                                       -DBUILD_CUDA_LIB:BOOL=OFF \
                                       -DBUILD_DOC:BOOL=OFF \
                                       -DBUILD_EXAMPLES:BOOL=OFF \
                                       -DBUILD_MATLAB_BINDINGS:BOOL=OFF \
                                       -DBUILD_PYTHON_BINDINGS:BOOL=OFF \
                                       -DBUILD_TESTS:BOOL=OFF \

        cd ${IMAGE_PROCESSING_BUILD_ROOT}/flann.OSID_${OS_ID}.buildtree.Release
        time cmake --build . -- install
    )
fi
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}${CMAKE_PREFIX_PATH:+:}${FLANN_IT}"
# echo "CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"

PCL_IT=${IMAGE_PROCESSING_BUILD_ROOT}/pcl.OSID_${OS_ID}.installtree.Release
if [[ -d "${PCL_IT}" ]]
then
    echo "PCL already in $PCL_IT"
else
    (
        if [[ ! -d pcl ]]
        then
            git clone -b feature_implement_pcl__SampleConsensusModelSphere_PointT___projectPoints https://github.com/fidergo-stephane-gourichon/pcl
            #https://github.com/PointCloudLibrary/pcl
            (
                # Workaround failure to build PCL on recent flann.
                #
                # bug
                # https://github.com/PointCloudLibrary/pcl/issues/804
                # fix
                # https://github.com/PointCloudLibrary/pcl/pull/3317
                # fix does not apply cleanly, just overwriting file.
                cd pcl
                curl https://raw.githubusercontent.com/PointCloudLibrary/pcl/ee0d8ce5fc644480e4cd5672cb440a731c6f3758/cmake/Modules/FindFLANN.cmake >| cmake/Modules/FindFLANN.cmake 
            )
        fi

        cd pcl
        export EXPECTED_KILOBYTES_OCCUPATION_PER_CORE=1200000
        "$IMAGE_PROCESSING_SOURCE_ROOT"/cmake_project_bootstrap.sh . ${MY_CMAKE_GENERATOR_OPTIONS:-} \
                                   -DCMAKE_BUILD_TYPE:STRING=Release \
                                   -DCMAKE_CXX_STANDARD=11 \
                                   -DWITH_QHULL=ON \
                                   -DWITH_VTK=ON \
                                   -DBUILD_visualization=ON

        cd ${IMAGE_PROCESSING_BUILD_ROOT}/pcl.OSID_${OS_ID}.buildtree.Release
        time cmake --build . -- install
    )
fi
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}${CMAKE_PREFIX_PATH:+:}${PCL_IT}"
# echo "CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"

IAGMM_BUILD_TYPE=Debug

IAGMM_IT=${IMAGE_PROCESSING_BUILD_ROOT}/IAGMM_Lib.OSID_${OS_ID}.installtree.${IAGMM_BUILD_TYPE}
if [[ -d "${IAGMM_IT}" ]]
then
    echo "IAGMM_Lib already in $IAGMM_IT"
else(
    if [[ ! -d IAGMM_Lib ]]
    then
        git clone https://github.com/robotsthatdream/IAGMM_Lib
    fi

    cd IAGMM_Lib
    export EXPECTED_KILOBYTES_OCCUPATION_PER_CORE=1100000
    "${IMAGE_PROCESSING_SOURCE_ROOT}"/cmake_project_bootstrap.sh . ${MY_CMAKE_GENERATOR_OPTIONS:-} \
                               -DCMAKE_BUILD_TYPE=${IAGMM_BUILD_TYPE} \

    cd ${IMAGE_PROCESSING_BUILD_ROOT}/IAGMM_Lib.OSID_${OS_ID}.buildtree.${IAGMM_BUILD_TYPE}
    time cmake --build . -- install
)
fi

CMAES_BUILD_TYPE=Debug

CMAES_IT=${IMAGE_PROCESSING_BUILD_ROOT}/libcmaes.OSID_${OS_ID}.installtree.${CMAES_BUILD_TYPE}
if [[ -d "${CMAES_IT}" ]]
then
    echo "libcmaes already in $CMAES_IT"
else(
    if [[ ! -d libcmaes ]]
    then
        #git clone https://github.com/beniz/libcmaes
        git clone https://github.com/robotsthatdream/libcmaes
    fi

    cd libcmaes
    export EXPECTED_KILOBYTES_OCCUPATION_PER_CORE=1000000
    "${IMAGE_PROCESSING_SOURCE_ROOT}"/cmake_project_bootstrap.sh . ${MY_CMAKE_GENERATOR_OPTIONS:-} \
                               -DCMAKE_BUILD_TYPE=${CMAES_BUILD_TYPE} \

    cd ${IMAGE_PROCESSING_BUILD_ROOT}/libcmaes.OSID_${OS_ID}.buildtree.${CMAES_BUILD_TYPE}
    time cmake --build . -- install
)
fi

export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}${CMAKE_PREFIX_PATH:+:}${CMAES_IT}"

IMAGE_PROCESSING_BUILD_TYPE=Debug

IMAGE_PROCESSING_IT=${IMAGE_PROCESSING_BUILD_ROOT}/image_processing.OSID_${OS_ID}.installtree.${IMAGE_PROCESSING_BUILD_TYPE}
if [[ -d "${IMAGE_PROCESSING_IT}" ]]
then
    echo "Image_Processing already in $IMAGE_PROCESSING_IT"
else(
#    if [[ ! -d image_processing ]]
#    then
#        git clone https://github.com/robotsthatdream/Lib_image_processing image_processing
#    fi

    cd "${IMAGE_PROCESSING_SOURCE_ROOT}"
    export EXPECTED_KILOBYTES_OCCUPATION_PER_CORE=2000000
    "${IMAGE_PROCESSING_SOURCE_ROOT}"/cmake_project_bootstrap.sh . ${MY_CMAKE_GENERATOR_OPTIONS:-} \
                               -DCMAKE_BUILD_TYPE=${IMAGE_PROCESSING_BUILD_TYPE} \
                               -DIAGMM_INSTALL_TREE:STRING="${IAGMM_IT}" \
                               -DCMAES_INSTALL_TREE:STRING="${CMAES_IT}" \

                               # *_INSTALL_TREE should be cleaned up.

    cd ${IMAGE_PROCESSING_SOURCE_ROOT}.OSID_${OS_ID}.buildtree.${IMAGE_PROCESSING_BUILD_TYPE}
    time cmake --build . -- install
)
fi

# FIXME limit core numbers depending on memory
