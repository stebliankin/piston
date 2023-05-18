#!/bin/bash

module load singularity-3.5.3

LIST="../lists/processed_masif_pos.txt"

mkdir -p ./logs
LOGS_DIR=./logs/prepare_test/
mkdir -p $LOGS_DIR

MAX_JOBS=1000

SLURM_ACC="your_acc"
SLURM_QOS="your_qos"
SLURM_NODE_TYPE="node_type"

while read p; do
    echo $p
    sbatch -J piston-prepare \
                    --account=$SLURM_ACC \
                    --qos=$SLURM_QOS \
                    -p $SLURM_NODE_TYPE \
                    -N 1 \
                    -n 1 \
                    -o $LOGS_DIR"/stdout-${p}-precompute.txt" \
                    -e $LOGS_DIR"/stderr-${p}-precompute.txt" \
                    ./preprocess_one_test.sh $p config_test_16R.py

         # If we exceed the limit of max jobs
     TOTAL_JOBS=$(($TOTAL_JOBS+1))
     echo "Job $TOTAL_JOBS"
    i=`squeue -u vsteb002 | wc -l`
    echo "Total jobs submitted: $i"
    if [ $i -gt $MAX_JOBS ]; then
            echo "Exceeded the job submission limit. Initiating Busy waiting..."
            while [ $i -gt $MAX_JOBS ]; do
                i=`squeue -u vsteb002 | wc -l`
                echo "Total jobs submitted: $i"
                sleep 1
            done
        fi
done < $LIST

LIST="../lists/processed_masif_neg.txt"

MAX_JOBS=1000

SLURM_ACC="iacc_giri"
SLURM_QOS="pq_giri"
SLURM_NODE_TYPE="investor"

while read p; do
    echo $p
    sbatch -J piston-prepare \
                    --account=$SLURM_ACC \
                    --qos=$SLURM_QOS \
                    -p $SLURM_NODE_TYPE \
                    -N 1 \
                    -n 1 \
                    -o $LOGS_DIR"/stdout-${p}-precompute.txt" \
                    -e $LOGS_DIR"/stderr-${p}-precompute.txt" \
                    ./preprocess_one_test.sh $p config_test_16R.py

         # If we exceed the limit of max jobs
     TOTAL_JOBS=$(($TOTAL_JOBS+1))
     echo "Job $TOTAL_JOBS"
    i=`squeue -u vsteb002 | wc -l`
    echo "Total jobs submitted: $i"
    if [ $i -gt $MAX_JOBS ]; then
            echo "Exceeded the job submission limit. Initiating Busy waiting..."
            while [ $i -gt $MAX_JOBS ]; do
                i=`squeue -u vsteb002 | wc -l`
                echo "Total jobs submitted: $i"
                sleep 1
            done
        fi
done < $LIST

LIST="../lists/masif_test_original.txt"
LOGS_DIR=./logs/prepare_test_original/
mkdir -p $LOGS_DIR


while read p; do
    echo $p
    sbatch -J piston-prepare \
                    --account=$SLURM_ACC \
                    --qos=$SLURM_QOS \
                    -p $SLURM_NODE_TYPE \
                    -N 1 \
                    -n 32 \
                    -o $LOGS_DIR"/stdout-${p}-precompute.txt" \
                    -e $LOGS_DIR"/stderr-${p}-precompute.txt" \
                    ./preprocess_one_test.sh $p config_test_16R_original.py

         # If we exceed the limit of max jobs
     TOTAL_JOBS=$(($TOTAL_JOBS+1))
     echo "Job $TOTAL_JOBS"
    i=`squeue -u vsteb002 | wc -l`
    echo "Total jobs submitted: $i"
    if [ $i -gt $MAX_JOBS ]; then
            echo "Exceeded the job submission limit. Initiating Busy waiting..."
            while [ $i -gt $MAX_JOBS ]; do
                i=`squeue -u vsteb002 | wc -l`
                echo "Total jobs submitted: $i"
                sleep 1
            done
        fi
done < $LIST
