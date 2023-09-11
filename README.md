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
    pip3 install ml_collections
    pip3 install kaleido
    pip3 install -U scikit-learn scipy matplotlib
    pip3 install pdb2sql


## Dataset

The datasets with pre-processed interface maps can be downloaded from Zenodo:

    # MaSIF-test:
    wget https://zenodo.org/record/7948337/files/masif_test.tar.gz?download=1 -O ./data/masif_test.tar.gz

    # CAPRI-score:
    wget https://zenodo.org/record/7948337/files/capri_score.tar.gz?download=1 -O ./data/capri_score.tar.gz

    # PDB-2023:
    wget https://zenodo.org/record/7948337/files/PDB_2023.tar.gz?download=1 -O ./data/PDB_2023.tar.gz


## Usage

* [PiSToN_test.ipynb](PiSToN_test.ipynb) contains an example of how to use the pre-trained PIsToN.
* [training_example](./training_example) provides an instructions of how to train PIsToN.

## Reference

Use the following reference to cite our work:

Stebliankin, V., Shirali, A., Baral, P. et al. Evaluating protein binding interfaces with transformer networks. Nat Mach Intell (2023). https://doi.org/10.1038/s42256-023-00715-4

## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: https://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png




