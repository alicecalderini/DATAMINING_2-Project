# Data Mining 2 – Project

**Academic Year 2025/2026**

## Project Description

This repository contains the work carried out for the Data Mining 2 course, focused on the analysis of the **Child Mind Institute (CMI) – Problematic Internet Use** Kaggle competition.

The goal is to develop predictive models capable of analyzing children's and adolescents' physical activity and fitness data, in order to identify early signs of problematic internet use. Specifically, the main task consists of predicting the **Severity Impairment Index (sii)**, a standard measure of problematic internet use.

🔗 [Kaggle Competition – Child Mind Institute](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data)

## Dataset

The project uses two datasets:

1. **Tabular Dataset** *(Modules 0, 1, 2)*
   A modified version of the original CMI dataset, with some original missing values altered and synthetic samples added.

2. **Time Series Dataset** *(Modules 0, 3)*
   Data collected from a wrist-worn sensor, containing acceleration measurements (x, y, z), the Euclidean Norm Minus One (ENMO) of the accelerometer signal, angle-Z, a non-wear binary flag, light, and battery voltage. The time series are associated with a subset of the original ids, each linked to the child's corresponding *sii*.

## Project Structure

The project is organized into **4 modules**:

### Module 0 – Data Understanding & Preparation
- **Tabular dataset**: exploration and analysis of the dataset's structure and key characteristics; data pre-processing (categorical variable encoding, feature scaling, missing value imputation); creation of new variables.
- **Time series dataset**: exploratory analysis and pre-processing in preparation for clustering, motif/anomaly detection, and classification tasks, using approximation techniques (SAX, PAA) if the dataset is too large.

### Module 1 – Advanced Data Preprocessing
- **Outlier Detection**: identification of the top 1% outliers using at least 3 methods from different families (density-based, angle-based, etc.), visualization in a 2D/3D scatter plot using dimensionality reduction techniques, and appropriately motivated outlier treatment.
- **Imbalanced Learning**: definition of a simple imbalanced binary classification task, solved with a Decision Tree or KNN, applying at least 2 imbalanced learning techniques (undersampling, oversampling) and discussing their impact on performance.

### Module 2 – Advanced ML & XAI
- **Advanced Classification**: multi-class classification task (predicting *sii*) using Logistic Regression, Support Vector Machines, Neural Networks, Ensemble Methods, and Gradient Boosting Machines, with hyperparameter tuning and evaluation through accuracy, precision, recall, F1-score, and ROC/PR curves.
- **Advanced Regression**: multiple regression task solved using 2 non-linear approaches, compared using appropriate metrics.
- **Explainability**: application of explanation methods (e.g., LIME, SHAP, LORE, Counterfactual Explainers) to one of the previous classification tasks.

### Module 3 – Time Series Analysis
- **Motifs/Discords**: identification of motifs and/or anomalies in the time series, with visualization and discussion of their relationship with shapelets.
- **Clustering**: application of at least 2 clustering algorithms on time series using an appropriate distance measure, with cluster analysis and visualization using at least 2 dimensionality reduction techniques.
- **Time Series Classification**: definition of one or more classification tasks solved using KNN (with at least 2 distances: Euclidean/Manhattan and DTW), Shapelets, and at least one additional method (Rocket, MUSE, CNN, RNN, etc.).
- **Sequential Pattern Mining** *(optional)*: discretization of the time series to perform sequential pattern mining and identify frequent patterns or trends.

## Repository Structure

\`\`\`
DATAMINING_2/
│
├── 0. DATA UNDERSTANDING/
├── 1. PREPROCESSING/
├── 2. ADVANCED ML & XAI/
├── 10. TIME SERIES/
│   ├── ... time series analysis notebooks
│   └── ... clustering files and results
│
└── README.md
\`\`\`
