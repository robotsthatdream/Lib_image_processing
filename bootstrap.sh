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

function select_ninja()
{
chmod -c a+x ninja_with_args.sh
export MY_CMAKE_GENERATOR_OPTIONS="-G Ninja -D CMAKE_MAKE_PROGRAM:STRING=\"$PWD/ninja_with_args.sh\" "
}

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

        cmake_project_bootstrap.sh . "${MY_CMAKE_GENERATOR_OPTIONS:-}" \
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
            git clone https://github.com/PointCloudLibrary/pcl
        fi

        cd pcl
        cmake_project_bootstrap.sh . "${MY_CMAKE_GENERATOR_OPTIONS:-}" \
                                   -DCMAKE_BUILD_TYPE:STRING=Release \
                                   -DCMAKE_CXX_STANDARD=11 . \
                                   -DWITH_QHULL=ON \
                                   -DWITH_VTK=ON \
                                   -DBUILD_visualization=ON

        cd ${IMAGE_PROCESSING_BUILD_ROOT}/pcl.OSID_${OS_ID}.buildtree.Release
        time cmake --build . -- install
    )
fi
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}${CMAKE_PREFIX_PATH:+:}${PCL_IT}"

IAGMM_IT=${IMAGE_PROCESSING_BUILD_ROOT}/IAGMM_Lib.OSID_${OS_ID}.installtree.Release
if [[ -d "${IAGMM_IT}" ]]
then
    echo "IAGMM_Lib already in $IAGMM_IT"
else(
    if [[ ! -d IAGMM_Lib ]]
    then
        git clone https://github.com/LeniLeGoff/IAGMM_Lib
    fi

    cd IAGMM_Lib
        cmake_project_bootstrap.sh . "${MY_CMAKE_GENERATOR_OPTIONS:-}" \
                               -DCMAKE_BUILD_TYPE=Release \

    cd ${IMAGE_PROCESSING_BUILD_ROOT}/IAGMM_Lib.OSID_${OS_ID}.buildtree.Release
    time cmake --build . -- install
)
fi

#export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}${CMAKE_PREFIX_PATH:+:}${IAGMM_IT}"

IMAGE_PROCESSING_IT=${IMAGE_PROCESSING_BUILD_ROOT}/image_processing.OSID_${OS_ID}.installtree.Release
if [[ -d "${IMAGE_PROCESSING_IT}" ]]
then
    echo "Image_Processing already in $IMAGE_PROCESSING_IT"
else(
#    if [[ ! -d image_processing ]]
#    then
#        git clone https://github.com/robotsthatdream/Lib_image_processing image_processing
#    fi

    cd "${IMAGE_PROCESSING_SOURCE_ROOT}"
    cmake_project_bootstrap.sh . "${MY_CMAKE_GENERATOR_OPTIONS:-}" \
                               -DCMAKE_BUILD_TYPE=Release \
                               -DIAGMM_INSTALL_TREE:STRING="${IAGMM_IT}" \


    cd ${IMAGE_PROCESSING_SOURCE_ROOT}.OSID_${OS_ID}.buildtree.Release
    time cmake --build . -- install
)
fi

# FIXME limit core numbers depending on memory
