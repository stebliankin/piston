#!/bin/bash

piston_PATH=$(pwd)/../../

PPI=$1

singularity exec ${piston_PATH}/env/piston.sif python3 $piston_PATH/piston.py  \
                                                                    --config config_test.py \
                                                                    prepare \
                                                                  --ppi $PPI \
                                                                  --no_download
