#!/bin/bash

piston_PATH=$(pwd)/../../

PPI=$1
CONFIG=$2

singularity exec ${piston_PATH}/env/piston.sif python3 $piston_PATH/piston.py  \
                                                                    --config $CONFIG \
                                                                    prepare \
                                                                  --ppi $PPI