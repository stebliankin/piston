# PDB-2023 Set

This repository provides instructions on how the PDB-2023 dataset was curated and processed. All scoring functions benchmarked in the accompanying paper were released before January 2023. Consequently, any complexes deposited after this date were not used in the training of any tool. The collection of protein complexes released in 2023 was manually curated and obtained from the RCSB PDB portal.

## 1. Obtaining Protein Complexes

Protein complexes were extracted using the following query from the PDB:

* (Oligomeric State IS "Hetero 2-mer") AND (Number of Distinct DNA Entities = 0)
 AND (Number of Distinct RNA Entities = 0) AND (Number of Distinct NA Hybrid Entities = 0)
AND (Release Date > 01/01/2023)

Repetitive complexes and protein-ligand interactions were manually removed from the dataset, resulting in a unique set of 53 hetero-dimers. All of the resultant PDB files are located in the [PDB](./PDB) folder.


## 2. Pre-processing

The following script was executed on a High-Performance Computing (HPC) machine with a SLURM job scheduler:

    ./1-preprocess.sh

Please note that this script utilizes a Singularity container named "_piston.sif_", which contains all the necessary dependencies. To create such a container, please refer to the [env](./../../env) folder older for detailed instructions.
