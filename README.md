# Threshold Analysis for Logistic Regression

This notebook demonstrates how changing the decision threshold affects classification performance in a logistic regression model trained on a synthetic binary dataset.

## Overview

The notebook:

- generates a synthetic classification dataset
- splits the data into train and test sets
- trains a logistic regression model
- computes class probabilities on the test set
- evaluates accuracy, error rate, precision, and recall across a range of thresholds
- visualizes how threshold choice impacts accuracy and error rate

## Notebook Contents

### 1. Data Preparation and Model Training

The first cell imports the required libraries, creates a synthetic dataset with `make_classification`, splits it into training and test sets, and fits a `LogisticRegression` model.

### 2. Threshold Sweep and Visualization

The second cell:

- evaluates thresholds from 0 to 1
- converts predicted probabilities into class labels using each threshold
- stores performance metrics in a pandas DataFrame
- plots accuracy and error rate across thresholds

## Requirements

Install the following Python packages:

- numpy
- pandas
- matplotlib
- scikit-learn

## How to Run

1. Open `gava.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
2. Run the first cell to generate the dataset and train the model.
3. Run the second cell to compute metrics and display the plot.

## Expected Output

The notebook produces a line chart showing:

- accuracy as the threshold changes
- error rate as the threshold changes
- a vertical marker at the default threshold of `0.5`

## Notes

- The dataset is synthetic, so results may vary if the random seed or model settings are changed.
- The notebook is designed to illustrate the tradeoff between threshold selection and classification performance.
