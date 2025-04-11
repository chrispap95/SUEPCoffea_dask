#!/bin/bash
LD_LIBRARY_PATH=$(pwd)/pythia/pythia8313/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
PYTHONPATH=$(pwd)/pythia/pythia8313/lib/:$PYTHONPATH
export PYTHONPATH
