# MRI-Segmentation
_Author: Antoine DELPLACE_  
_Last update: 17/01/2020_

This repository corresponds to the source code used for the MRI segmentation part of my Master Thesis entitled "__Segmentation and Generation of Magnetic Resonance Images by Deep Neural Networks__".

## Method description
The aim of the project is to achieve state-of-the-art performance in segmenting knee Magnetic Resonance Images (MRIs) thanks to a Neural Network architecture called __U-net__.

## Usage

### Dependencies
- Python 3.6.8
- Tensorflow 1.14
- Keras 2.2.4
- Numpy 1.16.2
- Pandas 0.24.2
- Matplotlib 3.0.3
- Scikit-image 0.15.0
- Scikit-learn 0.20.3

### File description
1. `main_unet.py` is the main file dedicated to training the model, saving the weights and plotting a comparison between the ground truth and the generated segmentation.

2. `test_boxplots.py` is the post-processing program responsible for the statistical analysis and the generation of boxplots.

## Results
The model demonstrates __state-of-the-art performance__ in segmenting bones and cartilages of knee MRIs. The hyperparameter tuning, the visual outputs and the qualitative results can be found in my Master thesis.

## References
1. A. Delplace. "Segmentation and Generation of Magnetic Resonance Images by Deep Neural Networks", _Master thesis at the University of Queensland_, October 2019. [arXiv:2001.05447](https://arxiv.org/abs/2001.05447)