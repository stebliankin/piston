#!/bin/bash


#cuda
#singularity shell --bind /opt,/disk,/usr/local/cuda/,/usr/lib64/,/usr/bin/ emomis.sif
#singularity shell --bind /usr/local/cuda/,/usr/lib64/,/usr/bin/ emomis.sif
#/bin/singularity shell --bind /usr/local/cuda/,/usr/lib64/,/usr/bin/ ./TransBind.sif
module load singularity-3.5.3
singularity shell ./piston.sif

#singularity shell --bind /opt,/disk masif_latest.sif

#singularity shell masif_latest.sif
