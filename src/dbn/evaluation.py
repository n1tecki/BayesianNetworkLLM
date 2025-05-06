from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix
)
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd


def predict_sepsis(patient_df, inference, LAB_COLS):
    out = {}

    for t, row in patient_df.reset_index(drop=True).iterrows():
        # only evidence from *this* slice – no gigantic dict:
        evidence_slice = {(lab, t): row[lab] for lab in LAB_COLS}

        q = inference.forward_inference(
            variables=[("sepsis", t)],
            evidence=evidence_slice
        )
        out[t] = q[("sepsis", t)].values[1]

        # If you want to keep a running filter, you can
        # evidence_so_far.update(evidence_slice)
    return out


def predict_df_data(df, inference, LAB_COLS, save_path="data/predictions.json"):
    predictions = {}
    test_ids = df.index.unique()
    for hadm_id in tqdm(test_ids):
        new_patient = df.loc[hadm_id].reset_index(drop=True)[LAB_COLS]
        pred = predict_sepsis(new_patient, inference, LAB_COLS)
        predictions[hadm_id] = pred

    with open(save_path, "w") as f:
        json.dump(predictions, f, indent=2)

    return predictions


def steps_to_threshold(sequence, threshold):
    """
    Return the index (1-based) of the first element in *sequence* that is
    >= *threshold*.  If the threshold is never met, return None.
    """
    for idx, value in enumerate(sequence, start=1):
        if value >= threshold:
            return idx
    return None


def compare_single_threshold(runs, dbn_thr=0.7, sofa_thr=2):
    """
    Given *runs* -- an iterable of (y_hat, y) tuples – return:
        earlier, later, equal     : counts
        deltas                    : list of (steps_yhat − steps_y)
    """
    earlier = later = equal = 0
    deltas = []

    for y_hat, y in runs:
        s_hat = steps_to_threshold(y_hat, dbn_thr)
        s_y   = steps_to_threshold(y,    sofa_thr)

        # ignore runs in which either series never hits its threshold
        if s_hat is None or s_y is None:
            continue

        delta = s_hat - s_y
        deltas.append(delta)

        if delta < 0:
            earlier += 1
        elif delta > 0:
            later   += 1
        else:
            equal   += 1

    plt.figure(figsize=(6, 4))
    plt.hist(deltas, bins=range(min(deltas), max(deltas)+2), edgecolor='black')
    plt.title(rf"""Δ steps to threshold ($y_{{\mathrm{{DBN}}}} - y_{{\mathrm{{SOFA}}}}$)
            Thresholds: $y_{{\mathrm{{DBN}}}}\geq{dbn_thr}$,  $y_{{\mathrm{{SOFA}}}}\geq{sofa_thr}$""")
    plt.xlabel("Steps difference")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    return earlier, later, equal, deltas


def compare_var_thresholds(runs,
                           yhat_thresholds=np.arange(0.5, 1.01, 0.01),
                           y_threshold=2):
    """
    For every threshold in *yhat_thresholds*, compute the proportion
    of runs in which ŷ is earlier / later / equal.
    Returns:
        thresholds, earlier%, later%, equal%
    """
    earlier_p = []
    later_p   = []
    equal_p   = []

    for thr in yhat_thresholds:
        earlier = later = equal = total = 0

        for y_hat, y in runs:
            s_hat = steps_to_threshold(y_hat, thr)
            s_y   = steps_to_threshold(y,  y_threshold)

            if s_hat is None or s_y is None:
                continue

            total += 1
            if   s_hat < s_y: earlier += 1
            elif s_hat > s_y: later   += 1
            else:             equal   += 1

        if total:
            earlier_p.append(earlier / total)
            later_p.append(later / total)
            equal_p.append(equal / total)
        else:
            earlier_p.append(np.nan)
            later_p.append(np.nan)
            equal_p.append(np.nan)

    plt.figure(figsize=(6, 4))
    plt.plot(thr, earlier_p, label=rf"$y_{{\mathrm{{DBN}}}} earlier")
    plt.plot(thr, later_p,  label="$y_{{\mathrm{{DBN}}}} later")
    plt.plot(thr, equal_p,  label="Tie")
    plt.title(rf"Proportion of outcomes vs. $y_{{\mathrm{{DBN}}}} threshold")
    plt.xlabel(rf"$y_{{\mathrm{{DBN}}}} threshold")
    plt.ylabel("Proportion of runs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return yhat_thresholds, earlier_p, later_p, equal_p


def binary_metrics(y_true, y_pred, *, label):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    ci_low, ci_up = proportion_confint(
        count=(y_true == y_pred).sum(),
        nobs=len(y_true),
        method="wilson"
    )

    return pd.Series({
        "accuracy":        acc,
        "acc_CI_low":      ci_low,
        "acc_CI_high":     ci_up,
        "sensitivity":     recall_score(y_true, y_pred),          # TPR
        "specificity":     recall_score(1 - y_true, 1 - y_pred),  # TNR
        "precision":       precision_score(y_true, y_pred),       # PPV
        "F1":              f1_score(y_true, y_pred),
        "MCC":             matthews_corrcoef(y_true, y_pred),
        "TP":              confusion_matrix(y_true, y_pred)[1, 1],
        "FP":              confusion_matrix(y_true, y_pred)[0, 1],
        "FN":              confusion_matrix(y_true, y_pred)[1, 0],
        "TN":              confusion_matrix(y_true, y_pred)[0, 0],
        "n":               len(y_true),
    }, name=label)


def compare_two_models(y_true, y_pred_a, y_pred_b,
                       label_a="DBN", label_b="SOFA", alpha=0.05):
    metrics_a = binary_metrics(y_true, y_pred_a, label=label_a)
    metrics_b = binary_metrics(y_true, y_pred_b, label=label_b)
    metrics_df = pd.concat([metrics_a, metrics_b], axis=1).T

    # 2×2 table for paired accuracy comparison
    both_correct     = ((y_pred_a == y_true) & (y_pred_b == y_true)).sum()
    a_only_correct   = ((y_pred_a == y_true) & (y_pred_b != y_true)).sum()
    b_only_correct   = ((y_pred_a != y_true) & (y_pred_b == y_true)).sum()
    both_wrong       = len(y_true) - (both_correct + a_only_correct + b_only_correct)

    table = [[both_correct, a_only_correct],
             [b_only_correct, both_wrong]]

    mcnemar_res = mcnemar(table, exact=False, correction=True, alpha=alpha)

    return metrics_df, mcnemar_res


def plot_accuracy_bars(metrics_df, title="Classification accuracy"):
    fig, ax = plt.subplots()
    xpos = np.arange(len(metrics_df))

    acc = metrics_df["accuracy"].values
    err_low = acc - metrics_df["acc_CI_low"].values
    err_up  = metrics_df["acc_CI_high"].values - acc
    ax.bar(xpos, acc, yerr=[err_low, err_up], capsize=6)

    ax.set_xticks(xpos)
    ax.set_xticklabels(metrics_df.index)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()