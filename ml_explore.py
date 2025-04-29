#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Microbiome Data Analysis and Machine Learning Pipeline

This script performs exploratory data analysis and machine learning on microbiome data.
It includes dimensionality reduction techniques (PCA, UMAP, PCoA) and classification models (SVM, Random Forest).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report
)
import umap
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa


def load_data(taxa_file, meta_file):
    """
    Load and preprocess the microbiome data and metadata.
    
    Args:
        taxa_file (str): Path to the taxa abundance data file
        meta_file (str): Path to the metadata file
        
    Returns:
        tuple: Processed feature matrix X and target labels Y
    """
    # Change working directory
    os.chdir(os.path.expanduser('~/JMW/mars2020'))
    
    # Load data
    df = pd.read_csv(taxa_file).T
    meta = pd.read_csv(meta_file)
    
    # Set the first row as the header
    df.columns = df.iloc[0]
    df.drop(df.index[0], inplace=True)
    
    # Extract unique individuals
    samples = df.index
    indv = set([x.split('.')[0] for x in samples])
    
    # Verify all individuals have metadata
    in_meta = [x in meta['participant'].values.astype(str) for x in indv]
    
    # Select first timepoint for each individual
    X = df.loc[[x + '.T.0' for x in indv]].copy()
    X.index = [x.split('.')[0] for x in X.index]
    
    # Extract target labels
    Y = np.array([meta['Cohort'][meta['SampleID'] == 'Participant ' + x + ' Sample 0'].values[0] for x in X.index])
    
    # Merge categories 'C' and 'D' into 'IBS'
    Y = ['IBS' if x == 'C' or x == 'D' else x for x in Y]
    
    # Remove instances with 'nan' labels
    drop_ind = [i for i in range(len(Y)) if Y[i] == 'nan']
    X = X.drop(X.index[drop_ind])
    Y = [Y[i] for i in range(len(Y)) if i not in drop_ind]
    Y = np.array(LabelEncoder().fit_transform(Y))
    
    # Remove uninformative features
    X = X.loc[:, X.sum() > 0]
    
    return X, Y


def perform_dimensionality_reduction(X, Y, output_dir='./'):
    """
    Perform and visualize dimensionality reduction techniques.
    
    Args:
        X (pd.DataFrame): Feature matrix
        Y (np.array): Target labels
        output_dir (str): Directory to save output plots
    """
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=Y, cmap='viridis')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Plot')
    plt.colorbar(scatter, label='Target')
    plt.savefig(f'{output_dir}pca.png')
    plt.close()
    
    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_result = reducer.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=Y, cmap='viridis', s=10)
    plt.colorbar(scatter)
    plt.title('UMAP Projection of the Data')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig(f'{output_dir}umap.png')
    plt.close()
    
    # PCoA
    X_numeric = X.apply(pd.to_numeric, errors='coerce').astype(float)
    bray_curtis_dm = beta_diversity(metric='braycurtis', counts=X_numeric.values, ids=X_numeric.index)
    pcoa_results = pcoa(bray_curtis_dm)
    pc1 = pcoa_results.samples['PC1']
    pc2 = pcoa_results.samples['PC2']
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pc1, pc2, c=Y, edgecolor='k')
    plt.xlabel(f"PC1 ({pcoa_results.proportion_explained[0]:.2%})")
    plt.ylabel(f"PC2 ({pcoa_results.proportion_explained[1]:.2%})")
    plt.title("PCoA Plot Using Bray-Curtis Dissimilarity")
    plt.savefig(f'{output_dir}pcoa.png')
    plt.close()
    
    return X_scaled


def train_svm_model(X, Y, test_size=0.3, random_state=42):
    """
    Train and evaluate an SVM model with grid search.
    
    Args:
        X (np.array): Scaled feature matrix
        Y (np.array): Target labels
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: Best SVM model, test data, and predictions
    """
    # Define parameter grid
    param_grid = {
        'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    }
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=Y
    )
    
    # Perform grid search
    grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5, scoring='balanced_accuracy')
    grid_search.fit(X_train, Y_train)
    
    # Print results
    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation balanced accuracy:", grid_search.best_score_)
    
    # Get best model and predictions
    best_svm = grid_search.best_estimator_
    Y_pred_best = best_svm.predict(X_test)
    
    # Evaluate model
    print("Accuracy with best model:", accuracy_score(Y_test, Y_pred_best))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_best))
    print("Classification Report:\n", classification_report(Y_test, Y_pred_best))
    
    return best_svm, X_train, X_test, Y_train, Y_test, Y_pred_best


def perform_feature_selection(X_train, X_test, Y_train, best_svm, n_features=30):
    """
    Perform feature selection using Recursive Feature Elimination.
    
    Args:
        X_train (np.array): Training features
        X_test (np.array): Test features
        Y_train (np.array): Training labels
        best_svm: Best SVM model from grid search
        n_features (int): Number of features to select
        
    Returns:
        tuple: Selected features for training and testing
    """
    selector = RFE(estimator=best_svm, n_features_to_select=n_features, step=1)
    selector = selector.fit(X_train, Y_train)
    
    X_train_rfe = selector.transform(X_train)
    X_test_rfe = selector.transform(X_test)
    
    return X_train_rfe, X_test_rfe, selector


def train_random_forest(X_train, X_test, Y_train, Y_test, n_estimators=5000, random_state=42):
    """
    Train and evaluate a Random Forest model.
    
    Args:
        X_train (np.array): Training features
        X_test (np.array): Test features
        Y_train (np.array): Training labels
        Y_test (np.array): Test labels
        n_estimators (int): Number of trees in the forest
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: Trained Random Forest model and predictions
    """
    # Initialize and train model
    rf = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, random_state=random_state)
    rf.fit(X_train, Y_train)
    
    # Get OOB predictions
    oob_pred_prob = rf.oob_decision_function_
    oob_pred = rf.classes_[oob_pred_prob.argmax(axis=1)]
    
    # Evaluate OOB performance
    f1 = f1_score(Y_train, oob_pred, average='weighted')
    precision = precision_score(Y_train, oob_pred, average='weighted')
    recall = recall_score(Y_train, oob_pred, average='weighted')
    accuracy = accuracy_score(Y_train, oob_pred)
    
    print(f"OOB F1 Score: {f1:.2f}")
    print(f"OOB Precision: {precision:.2f}")
    print(f"OOB Recall: {recall:.2f}")
    print(f"OOB Accuracy: {accuracy:.2f}")
    
    # Predict on test data
    Y_pred = rf.predict(X_test)
    
    # Evaluate test performance
    print("Test Accuracy:", accuracy_score(Y_test, Y_pred))
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
    
    return rf, Y_pred


def train_svm_with_selected_features(X_train, X_test, Y_train, Y_test, param_grid):
    """
    Train SVM model using selected features.
    
    Args:
        X_train (np.array): Training features after feature selection
        X_test (np.array): Test features after feature selection
        Y_train (np.array): Training labels
        Y_test (np.array): Test labels
        param_grid (dict): Parameter grid for grid search
        
    Returns:
        tuple: Best SVM model and predictions
    """
    grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5, scoring='balanced_accuracy')
    grid_search.fit(X_train, Y_train)
    
    best_svm_rfe = grid_search.best_estimator_
    Y_pred_best_rfe = best_svm_rfe.predict(X_test)
    
    print("Accuracy with best model after RFE:", accuracy_score(Y_test, Y_pred_best_rfe))
    print("Classification Report after RFE:\n", classification_report(Y_test, Y_pred_best_rfe))
    
    return best_svm_rfe, Y_pred_best_rfe


def perform_lasso_feature_selection(X_train, Y_train, random_state=42):
    """
    Perform feature selection using LassoCV.
    
    Args:
        X_train (np.array): Training features
        Y_train (np.array): Training labels
        random_state (int): Random seed for reproducibility
        
    Returns:
        LassoCV: Fitted LassoCV model
    """
    lasso_cv = LassoCV(alphas=np.logspace(-2, 5, 50), cv=5, random_state=random_state)
    lasso_cv.fit(X_train, Y_train)
    
    print(f"Optimal alpha: {lasso_cv.alpha_}")
    
    return lasso_cv


def main():
    """Main function to run the entire analysis pipeline."""
    # File paths
    taxa_file = "~/Downloads/Supplementary data I Microbiome data tables stool and biopsy.csv"
    meta_file = "~/JMW/mars2020/mars2020_merged.csv"
    
    # Load and preprocess data
    X, Y = load_data(taxa_file, meta_file)
    
    # Perform dimensionality reduction
    X_scaled = perform_dimensionality_reduction(X, Y)
    
    # Train SVM model
    param_grid = {
        'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    }
    best_svm, X_train, X_test, Y_train, Y_test, Y_pred_best = train_svm_model(X_scaled, Y)
    
    # Perform feature selection
    X_train_rfe, X_test_rfe, selector = perform_feature_selection(X_train, X_test, Y_train, best_svm)
    
    # Train Random Forest model
    rf, Y_pred_rf = train_random_forest(X_train_rfe, X_test_rfe, Y_train, Y_test)
    
    # Train SVM with selected features
    best_svm_rfe, Y_pred_svm_rfe = train_svm_with_selected_features(
        X_train_rfe, X_test_rfe, Y_train, Y_test, param_grid
    )
    
    # Perform Lasso feature selection
    lasso_cv = perform_lasso_feature_selection(X_train_rfe, Y_train)


if __name__ == "__main__":
    main()



