# IE4476: Dimensionality Reduction for Classification

This project compares Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) as preprocessing methods for image classification. It evaluates each reducer with 5-nearest neighbors and logistic regression, then saves accuracy curves, confusion matrices, classification reports, and a summary of the best configuration.

The default experiment uses the 70,000-sample MNIST dataset. A smaller scikit-learn Digits dataset option is also available for quicker local runs.

## Experiment pipeline

For every reducer, classifier, and requested output dimension, the script:

1. Loads and normalizes the selected dataset.
2. Creates a stratified 70/30 training and test split by default.
3. Standardizes the input features.
4. Fits PCA or LDA.
5. Trains kNN or logistic regression on the reduced features.
6. Measures test accuracy.
7. Saves an accuracy curve for each reducer/classifier pair.
8. Saves a confusion matrix and classification report for the best dimension.

### Methods

| Component | Configuration |
| --- | --- |
| PCA | Unsupervised reduction; defaults to 10, 20, 50, 100, 200, and 300 components. |
| LDA | Supervised reduction; defaults to 1 through 9 components for 10-class datasets. |
| kNN | `KNeighborsClassifier(n_neighbors=5)` |
| Logistic regression | `LogisticRegression(max_iter=2000)` |
| Split | Stratified 70% training and 30% testing with seed 42. |

## Recorded MNIST results

| Reducer | Classifier | Best dimension | Test accuracy |
| --- | --- | ---: | ---: |
| PCA | kNN | 50 | **95.64%** |
| PCA | Logistic regression | 300 | **91.95%** |
| LDA | kNN | 9 | **91.33%** |
| LDA | Logistic regression | 9 | **88.25%** |

PCA with 50 components and kNN produced the strongest recorded result. LDA is limited to at most `number of classes - 1` components but remains competitive with only nine dimensions.

## Setup

```bash
git clone https://github.com/Nixon080304/IE4476-Project-Dimensionality-Reduction-for-Classification.git
cd IE4476-Project-Dimensionality-Reduction-for-Classification

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn matplotlib
```

MNIST is downloaded from OpenML on the first run, so that experiment requires internet access. The Digits dataset ships with scikit-learn.

## Run

Run the default MNIST experiment:

```bash
python dim_red_classification.py
```

Run a faster experiment with the built-in Digits dataset:

```bash
python dim_red_classification.py \
  --dataset digits \
  --pca_dims 10 20 30 40 50 60 \
  --lda_dims 1 2 3 4 5 6 7 8 9
```

Select specific methods or change the split:

```bash
python dim_red_classification.py \
  --reducers PCA \
  --classifiers knn logreg \
  --pca_dims 20 50 100 \
  --test_size 0.2 \
  --seed 42 \
  --outdir outputs
```

## Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--dataset` | `mnist` | Dataset: `mnist` or `digits`. |
| `--reducers` | `PCA LDA` | Reduction methods to evaluate. |
| `--classifiers` | `knn logreg` | Classifiers to evaluate. |
| `--pca_dims` | `10 20 50 100 200 300` | PCA dimensions. |
| `--lda_dims` | `1 2 3 4 5 6 7 8 9` | LDA dimensions; invalid values are filtered. |
| `--test_size` | `0.3` | Fraction reserved for testing. |
| `--seed` | `42` | Split and PCA random seed. |
| `--outdir` | `outputs` | Base output directory. |

## Outputs

Each run creates a timestamped directory:

```text
outputs/<dataset>/<YYYYMMDD-HHMMSS>/
├── best_summary.txt
├── classification_report_<dataset>_<reducer>_<classifier>.txt
├── confusion_matrices/
│   └── <dataset>_<reducer>_<classifier>_cm.png
├── plots/
│   └── <dataset>_<reducer>_<classifier>.png
└── results.csv
```

The repository includes one complete MNIST result set under `outputs/mnist/20251101-194239`.

## Repository layout

```text
.
├── dim_red_classification.py # Full experiment and CLI
├── outputs/                  # Saved metrics, reports, and figures
└── README.md
```

## Implementation notes

- Input features are normalized before entering a pipeline with `StandardScaler`.
- PCA dimensions must not exceed the available sample and feature dimensions.
- LDA dimensions cannot exceed `number of classes - 1`.
- The unused helper `accuracyScore` duplicates scikit-learn's `accuracy_score`; experiments use the scikit-learn implementation.
- Full MNIST sweeps, especially kNN at several dimensions, can require substantial memory and runtime.

## References

1. X. Jiang, “Linear Subspace Learning-Based Dimensionality Reduction,” *IEEE Signal Processing Magazine*, 2011.
2. X. Jiang, “Asymmetric PCA and LDA,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2009.
3. X. Jiang, B. Mandal, and A. Kot, “Eigenfeature Regularization and Extraction,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2008.
4. Y. LeCun et al., “The MNIST Database of Handwritten Digits,” 1998.
5. F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, 2011.

## License

No license file has been added. Standard copyright restrictions apply unless the author grants additional permission.
