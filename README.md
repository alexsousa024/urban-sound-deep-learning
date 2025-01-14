# Urban Sound Classification with Neural Networks
# Overview
This project focuses on sound classification using the UrbanSound8K dataset. Two types of neural network models are implemented and trained:

Multilayer Perceptron (MLP)
Convolutional Neural Network (CNN)
The project is designed to classify urban sound categories effectively, leveraging TensorFlow for deep learning and Python for data preprocessing.

# Features
Data Preprocessing: Prepares audio data for training, including feature extraction and normalization.
Model Architectures:
MLP: A fully connected neural network for feature-based classification.
CNN: A convolutional neural network designed for processing spectrograms.
Evaluation: Uses metrics like accuracy and confusion matrices for model performance assessment.
Visualization: Graphs for accuracy, loss, and confusion matrix.
Utilities: Reusable utility functions for dataset loading, splitting, and preprocessing.

# Project Structure

project/
├── __init__.py                # Module initialization
├── MLP_TF.py                  # MLP classifier implementation
├── CNN_TF.py                  # CNN classifier implementation
├── utils.py                   # Utility functions for preprocessing
├── UrbanSound8K.csv           # Metadata for the UrbanSound8K dataset
├── project_realoficial.ipynb  # Main notebook with experiments
├── README.md                  # Project documentation

# Installation
Prerequisites
Python 3.8+
TensorFlow 2.x
Additional libraries: matplotlib, seaborn, numpy, scikit-learn



# Usage
Running the Notebook
Open the project_realoficial.ipynb notebook:
bash
Copiar código
jupyter lab
Execute the notebook cells to preprocess data, train models, and evaluate performance.
Using MLP



# Results
MLP: Provides baseline classification performance using handcrafted features.
CNN: Achieves superior results by leveraging spectrogram-based input.
Performance metrics include:

Training and validation accuracy
Confusion matrices for class-specific performance
Test set evaluation


# Acknowledgements
UrbanSound8K Dataset: Dataset Website
TensorFlow Documentation: TensorFlow.org
