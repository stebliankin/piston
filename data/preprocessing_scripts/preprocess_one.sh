#!/bin/bash

PPI=$1
piston_PATH=$(pwd)/../../

singularity exec ${piston_PATH}/env/piston.sif python3 $piston_PATH/piston.py  --config config.py \
                                                                    prepare \
                                                                  --ppi $PPI \
                                                                  --prepare_docking

rm -f /tmp/msms*

