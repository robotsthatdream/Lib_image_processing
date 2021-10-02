#!/bin/bash

set -eu

## Sanity check dependencies

TOOLS="cmake:cmake
/usr/include/eigen3:libeigen3-dev
git:git
/usr/include/boost/version.hpp:libboost-all-dev
/usr/include/flann/flann.h:libflann-dev
/usr/include/vtk*:libvtk6-dev
/usr/include/tbb/tbb.h:libtbb-dev
/usr/include/yaml-cpp/yaml.h:libyaml-cpp-dev
/usr/include/qhull/qhull.h:libqhull-dev
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

MISSING=""
for TOOLNP in ${TOOLS}
do IFS=: read FNAME PNAME <<< "$TOOLNP"
   echo -ne "Checking for $FNAME \011of $PNAME...  \0011"
   { which $FNAME || stat --format="%n" $( ls -1bd $FNAME || echo >&2 "$FNAME not found" )
   } || { MISSING="$MISSING $TOOLNP"
          echo "not found"
   }
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
        export EXPECTED_KILOBYTES_OCCUPATION_PER_CORE=600000
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
        fi

        cd pcl
        export EXPECTED_KILOBYTES_OCCUPATION_PER_CORE=2000000
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

CMM_IT=${IMAGE_PROCESSING_BUILD_ROOT}/CMM_Lib.OSID_${OS_ID}.installtree.Release
if [[ -d "${CMM_IT}" ]]
then
    echo "CMM_Lib already in $CMM_IT"
else(
    if [[ ! -d CMM_Lib ]]
    then
        git clone https://github.com/robotsthatdream/CMM_Lib
    fi

    cd CMM_Lib
    export EXPECTED_KILOBYTES_OCCUPATION_PER_CORE=2000000
    "$IMAGE_PROCESSING_SOURCE_ROOT"/cmake_project_bootstrap.sh . ${MY_CMAKE_GENERATOR_OPTIONS:-} \
                               -DCMAKE_BUILD_TYPE=Release \

    cd ${IMAGE_PROCESSING_BUILD_ROOT}/CMM_Lib.OSID_${OS_ID}.buildtree.Release
    time cmake --build . -- install
)
fi

#export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}${CMAKE_PREFIX_PATH:+:}${CMM_IT}"

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
    "$IMAGE_PROCESSING_SOURCE_ROOT"/cmake_project_bootstrap.sh . ${MY_CMAKE_GENERATOR_OPTIONS:-} \
                               -DCMAKE_BUILD_TYPE=${IMAGE_PROCESSING_BUILD_TYPE} \
                               -DCMM_INSTALL_TREE:STRING="${CMM_IT}" \


    cd ${IMAGE_PROCESSING_SOURCE_ROOT}.OSID_${OS_ID}.buildtree.${IMAGE_PROCESSING_BUILD_TYPE}
    time cmake --build . -- install
)
fi

# FIXME limit core numbers depending on memory
