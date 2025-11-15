#!/usr/bin/env python3
# Dimensionality Reduction on MNIST (PCA & LDA) with kNN + Logistic Regression
# Saves accuracy-vs-dimension plots, confusion matrices, and results.csv
import os, time, argparse, numpy as np, pandas as pd
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 140

def accuracyScore(y_true, y_pred):
    return (y_true == y_pred).mean()

def load_dataset(name: str):
    n = name.lower()
    if n == "mnist":
        m = fetch_openml("mnist_784", version=1, as_frame=False)
        X = m.data.astype("float32") / 255.0
        y = m.target.astype("int64")
        return X, y, {"name":"mnist","n_samples":len(y),"n_features":X.shape[1],"n_classes":len(np.unique(y))}
    elif n == "digits":
        d = load_digits()
        X, y = d.data.astype("float32")/16.0, d.target.astype("int64")
        return X, y, {"name":"digits","n_samples":len(y),"n_features":X.shape[1],"n_classes":len(np.unique(y))}
    else:
        raise ValueError("dataset must be mnist or digits")

def build_clf(name: str):
    n = name.lower()
    if n == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if n == "logreg":
        return LogisticRegression(max_iter=2000)
    raise ValueError("unknown classifier")

def run(dataset="mnist", reducers=("PCA","LDA"), classifiers=("knn","logreg"),
        pca_dims=(10,20,50,100,200,300), lda_dims=(1,2,3,4,5,6,7,8,9),
        test_size=0.3, seed=42, outbase="outputs"):
    X, y, info = load_dataset(dataset)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(outbase, info["name"], ts)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir,"plots"), exist_ok=True)
    os.makedirs(os.path.join(outdir,"confusion_matrices"), exist_ok=True)

    rows = []
    for red in reducers:
        redU = red.upper()
        dims = list(pca_dims if redU=="PCA" else lda_dims)
        if redU=="LDA":
            max_lda = info["n_classes"]-1
            dims = [d for d in dims if 1 <= d <= max_lda]
        for clf_name in classifiers:
            xs, ys = [], []
            best = {"acc":-1, "d":None, "pred":None}
            for d in dims:
                steps=[("scaler", StandardScaler())]
                if redU=="PCA": steps.append(("reducer", PCA(n_components=d, random_state=seed)))
                else:           steps.append(("reducer", LDA(n_components=d)))
                steps.append(("clf", build_clf(clf_name)))
                pipe = Pipeline(steps)
                pipe.fit(Xtr, ytr)
                yp = pipe.predict(Xte)
                acc = accuracy_score(yte, yp)
                rows.append({"dataset":info["name"],"reducer":redU,"classifier":clf_name,"n_components":d,"accuracy":acc})
                xs.append(d); ys.append(acc)
                if acc > best["acc"]:
                    best.update({"acc":acc,"d":d,"pred":yp,"pipe":pipe})
            # plot curve
            if xs:
                plt.figure()
                plt.plot(xs, ys, marker="o")
                plt.xlabel("n_components"); plt.ylabel("Accuracy")
                plt.title(f"{info['name']} — {redU} + {clf_name}")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir,"plots",f"{info['name']}_{redU}_{clf_name}.png"))
                plt.close()
            # confusion matrix for best
            if best["pred"] is not None:
                cm = confusion_matrix(yte, best["pred"])
                fig = plt.figure(); ax = fig.add_subplot(111)
                im = ax.imshow(cm, interpolation="nearest"); fig.colorbar(im)
                ax.set_title(f"Confusion — {info['name']} — {redU}+{clf_name} (best d={best['d']})")
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir,"confusion_matrices",f"{info['name']}_{redU}_{clf_name}_cm.png"))
                plt.close()
                with open(os.path.join(outdir,f"classification_report_{info['name']}_{redU}_{clf_name}.txt"),"w") as f:
                    f.write(classification_report(yte, best["pred"]))
    pd.DataFrame(rows).to_csv(os.path.join(outdir,"results.csv"), index=False)
    with open(os.path.join(outdir,"best_summary.txt"),"w") as f:
        if rows:
            df = pd.DataFrame(rows)
            for (r,c), sub in df.groupby(["reducer","classifier"]):
                s = sub.sort_values("accuracy", ascending=False).iloc[0]
                f.write(f"{r}+{c}: best d={int(s['n_components'])}, acc={s['accuracy']:.4f}\n")
    print(f"Done. See: {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist","digits"])
    ap.add_argument("--reducers", nargs="+", default=["PCA","LDA"])
    ap.add_argument("--classifiers", nargs="+", default=["knn","logreg"])
    ap.add_argument("--pca_dims", nargs="+", type=int, default=[10,20,50,100,200,300])
    ap.add_argument("--lda_dims", nargs="+", type=int, default=[1,2,3,4,5,6,7,8,9])
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()
    run(args.dataset, args.reducers, args.classifiers, args.pca_dims, args.lda_dims, args.test_size, args.seed, args.outdir)
