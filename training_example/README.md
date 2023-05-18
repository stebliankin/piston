# PIsToN training

This module includes instructions on how to train a new PIsToN model on a set of protein complexes.

## 1. Data pre-processing
Prior to model training, a set of positive and negative interface maps has to be pre-processed.
The first step includes generating acceptable and incorrect docking models from a list of native protein complexes.
Next, the physico-chemical features of interfaces from each docking model have to be projected into 2D multi-channel images. 

### Dependencies
A set of additional dependencies is required for docking and interface maps computation:

* [MaSIF](https://github.com/LPDI-EPFL/masif). The MaSIF installation guidlines can be found on [this link](https://github.com/LPDI-EPFL/masif/blob/master/docker_tutorial.md)
* [FireDock](http://bioinfo3d.cs.tau.ac.il/FireDock/). Use the following code for installation:

```
    wget http://bioinfo3d.cs.tau.ac.il/FireDock/download/fire_dock_download.zip
    unzip fire_dock_download.zip
    chmod -R 775 /FireDock/
    chmod +x /FireDock/buildFireDockParams.pl
    chmod +x /FireDock/PDBPreliminaries/prepareFragments.pl
    chmod +x /FireDock/PDBPreliminaries/PDBPreliminaries
```

* [HDOCKlite-v1.1](http://hdock.phys.hust.edu.cn/). The following link can be used to obtain HDODCK standalone verion: http://huanglab.phys.hust.edu.cn/software/hdocklite/

* [DSSP v.2.3.0](https://github.com/PDB-REDO/dssp). Use the following code for installation:

```
    wget "https://github.com/cmbi/dssp/archive/refs/tags/2.3.0.tar.gz"
    tar -zxvf 2.3.0.tar.gz
    cd dssp-2.3.0
    ./autogen.sh; ./configure; make; make install
    cd /
    rm -r dssp-2.3.0 2.3.0.tar.gz
```

### Example of computing interface maps
We provide a python wrapper to generate interface maps with [piston.py](https://github.com/stebliankin/piston/blob/main/piston.py)

```
    python3 piston.py prepare -h
    usage: piston.py prepare [-h] [--list LIST] [--ppi PPI] [--no_download]
                                [--download_only] [--prepare_docking]
    
    optional arguments:
      -h, --help         show this help message and exit
      --list LIST        List with PPIs in format PID_A_B
      --ppi PPI          PPI in format PID_A_B (mutually exclusive with the --list
                         option)
      --no_download      If set True, the pipeline will skip the download part.
      --download_only    If set True, the program will only download PDB
                         structures without processing them.
      --prepare_docking  If set True, re-dock native structures and pre-
                         process top 100 generated models
```

The script will automatically download complexes from [Protein Data Bank](https://www.rcsb.org/), transform them to a surface,
extract a pair of patches at the interface, compute all features, project it to images (interface maps), and save it as a numpy array.

We recommend using an HPC cluster for faster pre-processing.
An example of pre-processing on Slurm can be found in [piston/data/preprocessing_scripts/](../data/preprocessing_scripts/) of this repository.

### Configuration file
The required argument to ``piston.py`` is the path to a **configuration file** that should include a python dictionary ``config`` with the required parameters:

Input directories:
* `config['dirs']['data_prepare']` - the main directory for data preparation;
* `config['dirs']['raw_pdb']`  - directory with raw unprocessed PDB files

Directories with intermediate files (can be removed later to free up the space):
* `config['dirs']['protonated_pdb']` - directory with protonated PDB files with added hydrogens
* `config['dirs']['refined']` - PDB files refined with [FireDock](https://onlinelibrary.wiley.com/doi/10.1002/prot.21495)
* `config['dirs']['cropped_pdb']` - PDB files cropped at the interface
* `config['dirs']['chains_pdb']`- directory with PDB files sepparated by chains
* `config['dirs']['surface_ply']` - folder with protein surfaces in PLY format
* `config['dirs']['patches']` - extracted patches that has all [MaSIF features](https://github.com/LPDI-EPFL/masif)
* `config['dirs']['tmp']` - directory with temporary files

Output directory:
* `config['dirs']['grid']` - directory containing all pre-processed interface maps

Parameters:
* `config['ppi_const']['contact_d']` - minimum distance between residues to be considered as "contact point"
* `config['ppi_const']['surf_contact_r']` -minimum distance between surfaces to be considered as "contact point"
* `config['ppi_const']['patch_r']` - radius of the patch in angstroms
* `config['ppi_const']['crop_r']` - radius to crop
* `config['ppi_const']['points_in_patch']` - number of data points on a patch
* `config['mesh']['mesh_res']` - resolution of the mesh in angstroms

[config.py](../data/preprocessing_scripts/config.py) 
includes an example of the configuration file.

## 2. Training the model
The [piston_training.ipynb](./piston_training.ipynb) is the jupyter notebook that was used to train the final PIsToN network.
