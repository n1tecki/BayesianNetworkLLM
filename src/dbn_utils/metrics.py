import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix)



def metrics_vs_threshold_runs(runs, ground_truth, thresholds,
                              *, model="DBN",
                              zero_division=0, show=True):
    """
    Plot accuracy, precision, recall and F1 as a function of *stay-level*
    decision threshold.

    Parameters
    ----------
    runs : list of tuples
        Each item = (y_dbn, ts_dbn, y_sofa, ts_sofa) – exactly what you
        already store in `runs`.
    ground_truth : 1-d array-like of int
        One 0/1 label per stay (your `ground_truth` list).
    thresholds : iterable of float or int
        Thresholds to evaluate, e.g. np.arange(0.5,1.01,0.01) for DBN
        or np.arange(2,25,1) for SOFA totals.
    model : {"DBN","SOFA"}, default "DBN"
        Choose which score sequence in each run to use.
    zero_division : {0,1,"warn"}, default 0
        Passed through to precision/recall/F1.
    show : bool, default True
        Call plt.show() at the end.

    Returns
    -------
    pandas.DataFrame  (index = threshold)
        Columns = ["accuracy","precision","recall","F1"].
    """
    # pick the sequence position inside each run tuple
    seq_idx = 0 if model.upper() == "DBN" else 2
    y_true = np.asarray(ground_truth)

    acc, prec, rec, f1 = [], [], [], []

    for thr in thresholds:
        # stay-level prediction: “positive if ANY time point ≥ thr”
        y_pred = [int(any(v >= thr for v in run[seq_idx])) for run in runs]

        acc.append(accuracy_score(y_true, y_pred))
        prec.append(precision_score(y_true, y_pred,
                                    zero_division=zero_division))
        rec.append(recall_score(y_true, y_pred,
                                zero_division=zero_division))
        f1.append(f1_score(y_true, y_pred,
                           zero_division=zero_division))

    metrics_df = pd.DataFrame({
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "F1":        f1
    }, index=thresholds)

    # ---------------- plot -----------------
    plt.figure(figsize=(6, 4))
    plt.plot(metrics_df.index, metrics_df["accuracy"],  label="Accuracy")
    plt.plot(metrics_df.index, metrics_df["precision"], label="Precision")
    plt.plot(metrics_df.index, metrics_df["recall"],    label="Recall")
    plt.plot(metrics_df.index, metrics_df["F1"],        label="F1 score")

    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.title(f"{model}: metrics vs. threshold")
    plt.legend()
    plt.grid(axis="y")
    if show:
        plt.tight_layout()
        plt.show()

    return metrics_df