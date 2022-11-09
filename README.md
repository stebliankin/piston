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
We recommend creating a conda environment to prevent package overlaps:

    conda create -n piston python=3.7
    source activate piston

Next, install the following python packages:

    pip3 install tqdm
    pip3 install einops
    pip3 install keras-applications==1.0.8
    pip3 install opencv-python==4.5.5.62
    pip3 install pandas
    pip3 install torch==1.10.1
    pip3 install biopython --upgrade
    pip3 install plotly
    pip3 install torchsummary
    pip3 install torchsummaryX
    pip3 install scipy
    pip3 install sklearn
    pip3 install matplotlib
    pip3 install seaborn

## Dataset

The MaSIF-test dataset with pre-processed interface maps can be downloaded from the FIU server:

    wget https://users.cs.fiu.edu/~vsteb002/piston/masif_test.tar.gz

The docking models from the CAPRI score set can be obtained from the SBDB repository (https://data.sbgrid.org/dataset/843/):

    rsync -av rsync://data.sbgrid.org/10.15785/SBGRID/843 .

The pre-processed interface maps of a CAPRI score set can be downloaded as follows:

    wget https://users.cs.fiu.edu/~vsteb002/piston/capri_score.tar.gz


## Usage

Use the following link to download the pre-trained PIsToN model:

    wget https://users.cs.fiu.edu/~vsteb002/piston/PIsToN_multiAttn_contrast.pth


The `PiSToN_test.ipynb` contains an example of how to use the model.

## Reference

This repository is part of the paper "PIsToN: Evaluating Protein Binding Interfaces with Transformer Networks" and is currently under review at RECOMB 2023 conference.

Authors: Vitalii Stebliankin<sup>1</sup>, Azam Shirali<sup>1</sup>, Prabin Baral<sup>2</sup>, Prem Chapagain<sup>2,3</sup>, and
Giri Narasimhan<sup>1,3</sup>

[1] Bioinformatics Research Group (BioRG),
Knight Foundation School of Computing and Information Sciences,
Florida International University; 11200 SW 8th 10 St, Miami, 33199, USA
giri@cs.fiu.edu

[2] Department of Physics, College of Arts, Science and Education,
Florida International University; 11200 SW 8th St, Miami, 33199, USA

[3] Biomolecular Sciences Institute,
Florida International University; 11200 SW 8th St, Miami, 33199, USA
