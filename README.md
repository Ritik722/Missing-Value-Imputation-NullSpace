# Missing Value Imputation using Null Space (SVD)

This project implements a linear-algebra based approach to impute missing values by learning dependency constraints among features using the null space of the data matrix.

## Idea (high-level)
Given complete samples, we compute a null-space basis **A** such that:

A x = 0

These constraints represent linear relationships among attributes.  
For a row with missing values, we split variables into known and unknown parts and solve a linear system to recover missing entries.

## Features
- Builds null-space constraints using SVD (NumPy)
- Solves constraint equations to impute missing values
- Exports an imputed CSV

## Input format
A CSV file named `GTPvar.csv` with numeric columns (e.g., x1..x5).  
Missing entries should be blank/NaN.

## Data
This repository does not include the dataset.  
To run the script, add your own CSV file named `GTPvar.csv` in the same folder as `null_space_imputation.py`.

The CSV should contain numeric columns (e.g., x1..x5). Missing values should be blank/NaN.


## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
