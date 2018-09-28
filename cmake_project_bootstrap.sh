#!/bin/bash

# cmake_project_bootstrap.sh: one-step compile and install software
# locally.

# ========================================================================
# This is free software distributed under the MIT licence below.
#
#
# MIT License
#
# Copyright (c)2013-2017 Stéphane Gourichon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ========================================================================

# End of legalese, no on to work!

# Ask bash to abort on error, error on referring undefined variables, and error if part of a pipe fails.
set -eEuo pipefail

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

function show_usage_and_abort()
{
    cat <<EOF

========================================================================
What is this script?
------------------------------------------------------------------------
SUMMARY
------------------------------------------------------------------------
cmake_project_bootstrap.sh: one-step compile and install software
locally.
------------------------------------------------------------------------
DETAILS
------------------------------------------------------------------------
Input: one argument, path to a software source code tree that contains
a \`CMakeLists.txt\`.

Output:

* a separate compilation directory
* setup to install locally in a sibling directory.
* both are suffixed with the local operating system variant

Why?

* Time saver: this allows one-step building of software once you have
  the source code.

These benefits come from upstream tools, this scripts only provides
them easily:

* No privilege: installing on a local directory means no need to be
  root or whatever.

* Easy cleanup: installing on a local directory means easy to clean
  up.

* Easy save space: after install you can remove build tree (perhaps
  even source tree) easily.

* Suffixing with operating system name and version means that there's
  never any doubt about which operating system a particular directory
  is targetting.  Thus, no surprise about missing library at run time,
  no crash due to subtle library breakage.  If you change OS, just
  re-run the script on your new OS and bam it works!

[BAM! I got it running.](http://www.smbc-comics.com/comic/2011-02-17)

========================================================================

EOF

    exit 1
}

function error_block()
{
    echo "========================================================================"
    echo "ERROR: cmake_project_bootstrap.sh: cannot do its work."
    cat
    echo "========================================================================"
    echo "For explanations about this script run without argument:"
    echo "$0"
    exit 1
}

## One-way reference
#
# Wise man says: a built tree should refer to its source tree to
# enable updated builds, but a source tree must not know any build
# tree.  CMake and GNU configure do that naturally.
#
# Why?  Because one source tree can result in a variety of compiled
# binaries due to a variety of build settings, dependencies, compilers
# and operating systems.
#
# The fool builds binaries inside the source tree.  Variety comes and
# puts the fool into trouble.  Crying for simplicity, the fool gives
# up on keeping several binary artifacts from the same source and,
# even then, suffer on operating system upgrades.
#
# Thus, wisdom commands not to change anything in the source tree when
# building software.  If you don't change anything, the source tree
# cannot know if it's been compiled at all.  CMake and GNU configure
# do that naturally.
#
# As a bonus, it makes easy to delete big build tree (or even source
# tree) if only install tree is to be kept.
#
#
## Build and install tree close to source
#
# Wise man also says that test-drive-and-forget installations should
# have build tree and install trees close to the source tree.
#
# Why? It's the simplest thing to do in one automated step and make
# highly probable that build tree can find again source tree when an
# update is needed.
#
#
## Mark build and install tree with OS version
#
# Finally, wise man says that build and source tree should show what
# environment they target.  On operating system variation, is makes
# easy to know if any binary is compatible, and destroy any that is no
# longer useful.
#
# Wise man knowing its tools can reclaim space on OS upgrade with:
#
# find /mystorage -iname "*.OSID_myoldOS.*tree" -print0 | xargs -0 rm -rf

trap "echo See messages above for hints about what happened. | error_block" ERR


if [[ "$#" == 0 ]]
then
    {
        echo "ERROR: One argument needed: path to source tree."
        echo "SUGGESTION: If you mean current dir, just run like below:"
        echo "$0 ."
    } | error_block
fi

cd "$1"

if [[ ! -r CMakeLists.txt ]]
then
    echo "========================================================================"
    echo "ERROR: Not a CMake source tree."
    echo "Given path $1 lead to a location that does not contain a readable CMakeLists.txt:"
    echo "$PWD"
    echo "========================================================================"

    exit 1
fi

shift

SOURCEDIR_PARENT="${PWD%/}"

compute_OS_ID

# Stub left out because not all projects use git. Useful when you want to keep several install trees marked with exact version.
#PROJECT_REV_ID=$( cd "$SOURCEDIR_PARENT" ; git rev-parse --short HEAD )

MAYBE_CMAKE_BUILD_TYPE=$( echo "$@" | sed -r -n 's/^.*-D *CMAKE_BUILD_TYPE(|:STRING)=([^ ]*).*$/\2/p' )

CMAKE_BUILD_TYPE="${MAYBE_CMAKE_BUILD_TYPE:-RelWithDebInfo}"

SUFFIX=${CMAKE_BUILD_TYPE}

export PROJECT_BT="${SOURCEDIR_PARENT}.OSID_${OS_ID}.buildtree.${SUFFIX}"
export PROJECT_IT="${SOURCEDIR_PARENT}.OSID_${OS_ID}.installtree.${SUFFIX}"

# Assuming PROJECT_BT, it exists, is not rotten...

# GNU mkdir -p does not complain if already existing. If it fails on e.g. OSX please tell.
mkdir -p "${PROJECT_BT}"
cd "${PROJECT_BT}"

echo ========================================================================
echo ========================================================================
echo BUILD_DIRECTORY="$PWD"
echo ========================================================================
echo ========================================================================

# Actually call CMake.

(
    set -xv
# CFLAGS="-fsanitize=address -fno-omit-frame-pointer" CXXFLAGS="-fsanitize=address -fno-omit-frame-pointer" LDFLAGS="-fsanitize=address"
cmake -D CMAKE_BUILD_TYPE=${SUFFIX} -D CMAKE_INSTALL_PREFIX:PATH="${PROJECT_IT}" -D CMAKE_INSTALL_SYSCONFDIR="${PROJECT_IT}/etc" "$@" "${SOURCEDIR_PARENT}"
)


# Generate a script that ensures a build and install that will not
# miss any update.

cat >rebuild.sh <<EOF
#!/bin/bash

# This script rebuilds the project, keeping possible specific settings
# you may have made either with CMake or via extra arguments in the
# initial invocation of cmake_project_bootstrap.sh.

set -eu

cd -P "\$(dirname "\$(readlink -f "\$0")" )"
cmake --build . -- install
#make install

echo ========================================================================
echo ========================================================================
echo INSTALL_DIRECTORY="$PROJECT_IT"
echo ========================================================================
echo ========================================================================
EOF
chmod a+x rebuild.sh

if [[ -z "${NO_BUILD:-}" ]]
then

bash ./rebuild.sh

fi
