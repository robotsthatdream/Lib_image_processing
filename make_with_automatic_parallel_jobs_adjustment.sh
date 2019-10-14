#!/bin/bash

set -xv

PROCESSOR_COUNT=$( grep '^processor' /proc/cpuinfo  | wc -l )

echo "PROCESSOR_COUNT=$PROCESSOR_COUNT"

MAX_PARALLEL_JOBS_REQUESTED=$PROCESSOR_COUNT

if [[ -n "${EXPECTED_KILOBYTES_OCCUPATION_PER_CORE}" ]]
then
   AVAILABLE_KILOBYTES_AT_JOB_START=$( sed -n 's/^MemFree: *\([0-9]*\) kB/\1/p' /proc/meminfo )

   ACCEPTABLE_PARALLEL_JOBS_MEMORY_WISE=$(( AVAILABLE_KILOBYTES_AT_JOB_START / EXPECTED_KILOBYTES_OCCUPATION_PER_CORE ))

   echo "AVAILABLE_KILOBYTES_AT_JOB_START=$AVAILABLE_KILOBYTES_AT_JOB_START"
   echo "EXPECTED_KILOBYTES_OCCUPATION_PER_CORE=$EXPECTED_KILOBYTES_OCCUPATION_PER_CORE"
   echo "ACCEPTABLE_PARALLEL_JOBS_MEMORY_WISE=$ACCEPTABLE_PARALLEL_JOBS_MEMORY_WISE"

   MAX_PARALLEL_JOBS_REQUESTED=$(( $ACCEPTABLE_PARALLEL_JOBS_MEMORY_WISE > $PROCESSOR_COUNT ? $PROCESSOR_COUNT : $ACCEPTABLE_PARALLEL_JOBS_MEMORY_WISE ))
fi

if [[ "$MAX_PARALLEL_JOBS_REQUESTED" == "0" ]]
then
    echo >&2
    echo >&2 "=========================================================="
    echo >&2 "WARNING: LOW MEMORY, compilation may be slow or even fail."
    echo >&2 "=========================================================="
    echo >&2
else
    MY_EXTRA_ARGS="-j$MAX_PARALLEL_JOBS_REQUESTED"
fi


{
    echo "New run"
    echo "MY_EXTRA_ARGS=$MY_EXTRA_ARGS"
    echo "Args: $@"
} | tee -a compilation_memory_use.log >&2

exec /usr/bin/time --format='%M' --append --output=compilation_memory_use.log make "$MY_EXTRA_ARGS" "$@"
#-k ${BUILD_MAX_PARALLEL_PROCESSES:-${PROCESSOR_COUNT}} "$@"
