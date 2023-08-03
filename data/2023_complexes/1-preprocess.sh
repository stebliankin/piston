#!/bin/bash

module load singularity-3.5.3

mkdir -p ./logs
LOGS_DIR=./logs/perpare/
mkdir -p $LOGS_DIR

SLURM_ACC="iacc_giri"
SLURM_QOS="pq_giri"
SLURM_NODE_TYPE="investor"


#while read p; do
#    echo $p

sbatch -J PIsToN -a 0-65 \
                --account=$SLURM_ACC \
                --qos=$SLURM_QOS \
                -p $SLURM_NODE_TYPE \
                -N 1 \
                -n 1 \
                -o $LOGS_DIR"/stdout-precompute.txt" \
                -e $LOGS_DIR"/stderr-precompute.txt" \
                ./preprocess_one.sh
#
#         # If we exceed the limit of max jobs
#     TOTAL_JOBS=$(($TOTAL_JOBS+1))
#     echo "Job $TOTAL_JOBS"
#    i=`squeue -u vsteb002 | wc -l`
#    echo "Total jobs submitted: $i"
#    sleep 1
#    if [ $i -gt $MAX_JOBS ]; then
#            echo "Exceeded the job submission limit. Initiating Busy waiting..."
#            while [ $i -gt $MAX_JOBS ]; do
#                i=`squeue -u vsteb002 | wc -l`
#                echo "Total jobs submitted: $i"
#                sleep 3
#            done
#        fi
#done < $LIST

