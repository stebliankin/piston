# PIsToN
PIsToN (evaluating <b>P</b>rotein Binding <b>I</b>nterface<b>s</b> with <b>T</b>ransf<b>o</b>rmer <b>N</b>etworks) - 
is a novel deep learning-based approach for distinguishing native-like protein complexes from decoys. 
Each protein interface is transformed into a collection of 2D images (interface maps), 
where each image corresponds to a geometric or biochemical property in which pixel intensity represents the feature values.
Such data representation provides atomic-level resolution of relevant protein characteristics. 
To build hybrid machine learning models, additional empirical-based energy terms are computed and provided as inputs to the neural network.
The model is trained on thousands of native and computationally predicted protein complexes that contain challenging examples.
The multi-attention transformer network is also endowed with explainability by highlighting the specific features and binding sites that were the most important for the classification decision.

## Installation
For an optimal setup, we suggest using a conda environment to avoid package conflicts. Set it up as follows:

    conda create -n piston python=3.7
    source activate piston

Next, install the following python packages:

    pip3 install \
    tqdm \
    einops \
    keras-applications==1.0.8 \
    opencv-python==4.5.5.62 \
    pandas \
    torch==1.10.1 \
    biopython --upgrade \
    plotly \
    torchsummary \
    torchsummaryX \
    scipy \
    sklearn \
    matplotlib \
    seaborn \
    ml_collections \
    kaleido \
    -U scikit-learn \
    pdb2sql

As an alternative, you can fetch our Singularity container which comes with pre-configured dependencies:

    wget https://users.cs.fiu.edu/~vsteb002/piston_sif/piston.sif

| Note: The container was built with Singularity v3.5.3.

## Usage

PIsToN is designed to evaluate the interfaces of protein complexes. 
It offers two primary modules: "prepare" and "infer".

    piston -h
    usage: PIsToN [-h] [--config CONFIG] {prepare,infer} ...
    
    positional arguments:
      {prepare,infer}
        prepare        Data preparation module
        infer          Inference module
    
    optional arguments:
      -h, --help       show this help message and exit
      --config CONFIG  config file

### Inference module

The "infer" module computes PIsToN scores for protein complexes and visualizes the associated interface maps.

    usage: PIsToN infer [-h] --pdb_dir PDB_DIR [--list LIST] [--ppi PPI] --out_dir
                        OUT_DIR
    
    optional arguments:
      -h, --help         show this help message and exit
      --pdb_dir PDB_DIR  Path to the PDB file of the complex that we need to
                         score.
      --list LIST        Path to the list of protein complexes that we need to
                         score. The list should contain the PPIs in the following
                         format: PID_ch1_ch2, where PID is the name of PDB file,
                         ch1 is the first chain(s) of the protein complex , and
                         ch2 is the second chain(s). Ensure that PID does not
                         contain an underscore
      --ppi PPI          PPI in format PID_A_B (mutually exclusive with the --list
                         option)
      --out_dir OUT_DIR  Directory with output files.

The folder [example](./example) contains an example of running "infer" on two proteins: 6xe1AB-delRBD-100ns and 6xe1AB-wtRBD-100ns.
The proteins correspond to SARS-CoV-2 RBD/antibody complexes (delta variant and wild type) after 100ns of MD simulations
(see the study of [Baral et al.](https://doi.org/10.1016/j.bbrc.2021.08.036) for details).
The output is organized as follows:

- **PIsToN_scores.csv** - PIsToN scores in CSV format (Note: Lower scores indicate better binding).
- **gird_16R** - Directory containing interface maps in numpy format.
- **intermediate_files** - Intermediate files including proteins after prototantion, side-chain refinement, cropping, triangulation, and patch extraction.
- **patch_vis** - HTML files with interactive visualization of interface maps.

The "prepare" module facilitates the pre-computation of interface maps for extensive datasets.
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

If processing large datasets, we recommend running
We recommend using an HPC cluster for faster pre-processing.
An example of pre-processing on Slurm can be found in [piston/data/preprocessing_scripts/](../data/preprocessing_scripts/) of this repository.


## Benchmarks

The notebook [PiSToN_test.ipynb](PiSToN_test.ipynb) can be used to replicate the results reported in our paper.

## Training

[training_example](./training_example) provides an instructions of how to train PIsToN.

## Reference

Use the following reference to cite our work:

Stebliankin, V., Shirali, A., Baral, P. et al. Evaluating protein binding interfaces with transformer networks. Nat Mach Intell (2023). https://doi.org/10.1038/s42256-023-00715-4

## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: https://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png




