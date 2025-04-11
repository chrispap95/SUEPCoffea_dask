#!/bin/bash
# Original version is 106
# Use 107 for latest Pythia8 with proper python support

version=$1
if [ -z "$version" ]; then
    version=106
fi

LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_${version}/x86_64-el9-gcc13-opt/
source "$LCG"/setup.sh

# shellcheck disable=SC1091
source myenv/bin/activate
