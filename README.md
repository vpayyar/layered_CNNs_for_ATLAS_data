# Introduction
This repository contains codes used to train 2D layered CNNs for performing signal-background classification on ATLAS RPV-SUSY data. 

Tools for use for other datasets:
1. Data visulizaiton
2. Training CNN
3. Viewing model results: learning curves and predictions

# Dataset
The dataset consists of simulation results for the ATLAS experiment. The signal is RPV-SUSY and background is QCD.
For further information refer: https://arxiv.org/abs/1711.03573

# Further details:
- To see plots of raw data, use the jupyter notebook `data_visualization/1_view_data.ipynb`
- The code containing codes to train data `atlas_cnn/main_code/code/`
- Code that trains a set of CNNs on ATLAS SUSY data
The folder `main_code/jpt_notebooks/` : contains Jupyter notebooks that can read models and plot roc curves. 

For example, `main_code/jpt_notebooks/2_cnn_yaml_config.ipynb` has widgets to achieve this.

The notebook `main_code/jpt_notebooks/2_cnn_yaml_config.ipynb` performs full training and testing.
