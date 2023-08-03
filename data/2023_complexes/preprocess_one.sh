#!/bin/bash

PIsToN_PATH=$(pwd)/../../

IFS=$'\n' read -d '' -r -a ALL_PDBS < ./list_complexes.txt
PDB_ID=${ALL_PDBS[${SLURM_ARRAY_TASK_ID}]}

singularity exec ${PIsToN_PATH}/env/piston.sif python3 $PIsToN_PATH/piston.py  --config config16R.py \
                                                                    prepare \
                                                                  --ppi $PDB_ID \
                                                                  --prepare_docking

rm -f /tmp/msms*
