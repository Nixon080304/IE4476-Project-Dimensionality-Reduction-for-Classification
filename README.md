IE4476 Project – Dimensionality Reduction for Classification
============================================================

This repository contains the implementation and experiments for the IE4476 course project 
“Dimensionality Reduction for Classification”, focusing on how PCA and LDA affect classification 
performance on the MNIST handwritten digits dataset.

The project applies:
- Principal Component Analysis (PCA) – unsupervised, variance-preserving dimensionality reduction
- Linear Discriminant Analysis (LDA) – supervised, class-separating dimensionality reduction
- Classifiers: kNN and Logistic Regression


Dataset
============================================================

The MNIST dataset is used in this project. It contains:
- 70,000 handwritten digit images
- Resolution 28×28 pixels (flattened into 784 features)
- 10 classes (digits 0–9)

Preprocessing:
- Loaded via sklearn `fetch_openml('mnist_784')`
- Converted to float32
- Normalized to [0,1]
- Split into 70% training and 30% testing
- Stratified sampling maintains class balance
- Random seed = 42 for reproducibility


Dimensionality Reduction Methods
============================================================

--------------------
1. Principal Component Analysis (PCA)
--------------------
- Unsupervised learning method.
- Finds orthogonal directions (principal components) that maximize variance.
- Based on eigen-decomposition of the total scatter (covariance) matrix.
- Retains top-k eigenvectors with largest eigenvalues.

PCA Dimensions evaluated:
10, 20, 50, 100, 200, 300

--------------------
2. Linear Discriminant Analysis (LDA)
--------------------
- Supervised method using class labels.
- Maximizes the ratio:
    between-class scatter / within-class scatter
- Produces features that maximize class separability.
- Maximum number of LDA components = C − 1 = 9 (MNIST has 10 classes).

LDA Dimensions evaluated:
1, 2, 3, 4, 5, 6, 7, 8, 9


Classifiers
============================================================

Two classifiers were applied after dimensionality reduction:

1. k-Nearest Neighbors (kNN)
   - k = 5
   - Distance-based, non-parametric classifier

2. Logistic Regression
   - max_iter = 2000
   - Linear classifier with L2 regularization


Experimental Pipeline
============================================================

For each combination of:
- Reducer (PCA or LDA)
- Classifier (kNN or Logistic Regression)
- Dimensionality (various values)

The code performs:

1. Standardization using StandardScaler
2. Apply PCA or LDA with chosen number of components
3. Train classifier on reduced feature space
4. Predict test labels
5. Measure accuracy
6. Save:
   - accuracy vs. dimensionality plot
   - confusion matrix image
   - classification report (precision, recall, F1)
   - results.csv summarizing all experiments
   - best_summary.txt listing best results for each method


Results
============================================================

Best test accuracies:

| Method | Classifier | Best Dim | Accuracy |
|--------|------------|----------|----------|
| PCA    | kNN        | 50       | 95.64%   |
| PCA    | LogReg     | 300      | 91.95%   |
| LDA    | kNN        | 9        | 91.33%   |
| LDA    | LogReg     | 9        | 88.25%   |

Observations:
- PCA + kNN achieved the highest accuracy (95.64% at 50 components).
- Logistic Regression performs best with higher PCA dimensions.
- LDA, although limited to 9 components, still performs competitively due to strong discriminative power.


References
============================================================

[1] X. Jiang, "Linear Subspace Learning-Based Dimensionality Reduction," IEEE Signal Processing Magazine, 2011.
[2] X. Jiang, "Asymmetric PCA and LDA," IEEE TPAMI, 2009.
[3] X. Jiang, B. Mandal, and A. Kot, "Eigenfeature Regularization and Extraction," IEEE TPAMI, 2008.
[4] Y. LeCun, "MNIST Database," 1998.
[5] Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, 2011.
