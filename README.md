# PCA from Scratch and Machine Learning Performance Analysis

## Overview

This project demonstrates the implementation of Principal Component Analysis (PCA) from scratch using NumPy and compares it with Scikit-learn's PCA implementation. The project investigates how dimensionality reduction affects machine learning model performance and provides interactive visualizations through a Streamlit dashboard.

## Project Structure

```text
.
├── PCA_.ipynb          # Complete notebook implementation
├── app.py             # Interactive Streamlit dashboard
├── requirements.txt   # Project dependencies
└── README.md
```

## Features

### PCA from Scratch

* Mean centering
* Covariance matrix computation
* Eigenvalue and eigenvector calculation
* Principal component selection
* Data projection into lower dimensions

### Scikit-Learn PCA

* PCA implementation using Scikit-learn
* Explained variance analysis
* Scree plot visualization
* Component comparison

### Machine Learning Models

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest

### Performance Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Interactive Streamlit Dashboard

* Adjustable number of principal components
* PCA visualization
* Feature contribution analysis
* Model comparison before and after PCA
* Explained variance plots
* Runtime comparison

## Dataset

The project uses the Breast Cancer Wisconsin Dataset from Scikit-learn.

Dataset Characteristics:

* 569 samples
* 30 features
* Binary classification problem

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Notebook

Open the notebook:

```bash
jupyter notebook PCA_.ipynb
```

or upload the notebook to Google Colab.

## Running the Streamlit Application

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser.

## Visualizations Included

* Dataset Overview
* Feature Correlation Heatmap
* PCA Scatter Plot
* Explained Variance Curve
* Scree Plot
* Feature Contribution Matrix
* Model Performance Comparison
* Confusion Matrix
* PCA Runtime Comparison

## PCA Mathematical Foundation

PCA is based on eigenvalue decomposition of the covariance matrix.

Characteristic Equation:

[
\det(A - \lambda I) = 0
]

Eigenvector Equation:

[
Av = \lambda v
]

Projection Formula:

[
Z = XW
]

where:

* (X) is the standardized dataset
* (W) is the matrix of principal components
* (Z) is the transformed dataset

## Results

The project evaluates how dimensionality reduction affects classification performance and identifies the trade-off between information retention and computational efficiency.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

## Author

Fethulmubin Ahmed

Electrical and Computer Engineering Student

Addis Ababa University
