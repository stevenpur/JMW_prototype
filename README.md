# IBS Microbiome Profile Prediction

A machine learning prototype for predicting Irritable Bowel Syndrome (IBS) using microbiome data. This project implements various ML approaches to analyze and classify microbiome profiles for potential IBS diagnosis.

## Features
- Data preprocessing and feature engineering
- Dimensionality reduction (PCA, UMAP, PCoA)
- Classification models (SVM, Random Forest)
- Feature selection (RFE, Lasso)
- Interactive visualization (R Shiny)

## Implementation
- `ml_explore.py`: Main analysis pipeline
- `shiny.R`: Interactive data visualization
- Data files: Microbiome data tables and sample information

## Requirements
- Python: scikit-learn, pandas, numpy, matplotlib, umap-learn, scikit-bio
- R: Shiny package

## Usage
1. Install required packages
2. Run analysis: `python ml_explore.py`
3. Launch visualization: `R shiny.R`

## Results
Models demonstrate potential for IBS prediction using microbiome profiles, evaluated through accuracy, precision, recall, and F1 scores.

## Future Work
- Integration of additional microbiome features
- Exploration of deep learning approaches
- Validation on larger datasets
- Clinical translation studies
