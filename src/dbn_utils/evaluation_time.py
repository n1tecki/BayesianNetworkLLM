"""
Evaluation utilities - step- and time-aware.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix)
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def steps_to_threshold(sequence, threshold):
    """1-based index of first value ≥ threshold, else None."""
    for idx, v in enumerate(sequence, start=1):
        if v >= threshold:
            return idx
    return None


def time_to_threshold(sequence, timestamps, threshold):
    """Timestamp of first value ≥ threshold, else None."""
    for v, ts in zip(sequence, timestamps):
        if v >= threshold:
            return ts
    return None


# ---------------------------------------------------------------------------
# timing-aware comparison
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def compare_single_threshold(runs, dbn_thr=0.7, sofa_thr=2, show_plots=True):
    """
    Compare lead-times for one fixed pair of thresholds.

    Each run: (y_dbn, ts_dbn, y_sofa, ts_sofa)
    Returns counts and the raw Δ lists (steps and hours).
    """
    earlier = later = equal = 0
    deltas_steps, deltas_hours = [], []

    for y_dbn, ts_dbn, y_sofa, ts_sofa in runs:
        idx_dbn  = steps_to_threshold(y_dbn,  dbn_thr)
        idx_sofa = steps_to_threshold(y_sofa, sofa_thr)
        t_dbn    = time_to_threshold(y_dbn,  ts_dbn,  dbn_thr)
        t_sofa   = time_to_threshold(y_sofa, ts_sofa, sofa_thr)

        if idx_dbn is None or idx_sofa is None:
            continue  # one model never fired

        step_delta = idx_dbn - idx_sofa
        time_delta = (t_dbn - t_sofa).total_seconds() / 3600.0  # hours

        deltas_steps.append(step_delta)
        deltas_hours.append(time_delta)

        if step_delta < 0: earlier += 1
        elif step_delta > 0: later += 1
        else: equal += 1

    # --- steps histogram (separate plot) ---
    if show_plots and deltas_steps:
        bins_steps = range(min(deltas_steps), max(deltas_steps) + 2)
        plt.figure(figsize=(7, 4))
        plt.hist(deltas_steps, bins=bins_steps, edgecolor="black", color="#8ca6c8", alpha=1.0)
        plt.title("Δ labs (DBN − SOFA)")
        plt.xlabel("Labs")
        plt.ylabel("Frequency")
        plt.xticks(list(bins_steps))
        plt.tight_layout()
        plt.show()

        # --- hours histogram (separate plot) ---
        max_hours = 500
        delta_h_clip = np.clip(deltas_hours, -max_hours, max_hours)
        bins_h = np.arange(-max_hours, max_hours + 1, 1)
        plt.figure(figsize=(7, 4))
        plt.hist(delta_h_clip, bins=bins_h, edgecolor="black", color="#8ca6c8", alpha=1.0)
        plt.axvline(0, ls="--", lw=0.8)
        plt.title("Δ hours")
        plt.xlabel("Hours")
        plt.ylabel("Frequency")
        plt.xlim(-50, 50)
        plt.xticks(np.arange(-50, 51, 3))
        plt.tight_layout()
        plt.show()
        #ax[1].set_xlim(-max_hours, max_hours)

        # --- panel 2: minutes, ±10 000 min, 1-min bins ------------
        # delta_m_clip = np.clip(delta_min, -max_minutes, max_minutes)
        # bins_m = np.arange(-max_minutes, max_minutes + 1, 1)
        # ax[2].hist(delta_m_clip, bins=bins_m, edgecolor="black")
        # ax[2].axvline(0, ls="--", lw=0.8)
        # ax[2].set_title("Δ minutes")
        # ax[2].set_xlabel("Minutes")
        # ax[2].set_ylabel("Frequency")
        # ax[2].set_xlim(-max_minutes, max_minutes)      # ← and this one

        # plt.tight_layout()
        # plt.show()

    return earlier, later, equal, deltas_steps, deltas_hours



# ---------------------------------------------------------------------------
# threshold sweep
# ---------------------------------------------------------------------------

def compare_var_thresholds(runs,
                           yhat_thresholds=np.arange(0.5, 1.01, 0.01),
                           y_threshold=2):
    """
    For every DBN threshold compute the proportion of runs in which DBN fires
    earlier / later / tie (by steps).  Uses runs where both models fire.
    """
    earlier_p, later_p, equal_p = [], [], []

    for thr in yhat_thresholds:
        earlier = later = equal = total = 0

        for y_hat, _ts_hat, y, _ts_sofa in runs:

            s_hat = steps_to_threshold(y_hat, thr)
            s_y   = steps_to_threshold(y,    y_threshold)

            if s_hat is None or s_y is None:
                continue

            total += 1
            if   s_hat < s_y: earlier += 1
            elif s_hat > s_y: later   += 1
            else:             equal   += 1

        if total:
            earlier_p.append(earlier / total)
            later_p.append(later   / total)
            equal_p.append(equal   / total)
        else:
            earlier_p.append(np.nan)
            later_p.append(np.nan)
            equal_p.append(np.nan)

    plt.figure(figsize=(6, 4))
    plt.plot(yhat_thresholds, earlier_p, label="DBN earlier")
    plt.plot(yhat_thresholds, later_p,  label="DBN later")
    plt.plot(yhat_thresholds, equal_p,  label="Tie")
    plt.title("Proportion of outcomes vs. DBN threshold")
    plt.xlabel("DBN probability threshold")
    plt.ylabel("Proportion")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return yhat_thresholds, earlier_p, later_p, equal_p


# ---------------------------------------------------------------------------
# metrics & McNemar
# ---------------------------------------------------------------------------

def binary_metrics(y_true, y_pred, *, label):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    ci_low, ci_up = proportion_confint(
        count=(y_true == y_pred).sum(), nobs=len(y_true), method="wilson"
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return pd.Series({
        "accuracy":    acc,
        "acc_CI_low":  ci_low,
        "acc_CI_high": ci_up,
        "sensitivity": recall_score(y_true, y_pred),
        "specificity": recall_score(1 - y_true, 1 - y_pred),
        "precision":   precision_score(y_true, y_pred),
        "F1":          f1_score(y_true, y_pred),
        "MCC":         matthews_corrcoef(y_true, y_pred),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "n": len(y_true)
    }, name=label)


def compare_two_models(y_true, y_pred_a, y_pred_b,
                       label_a="DBN", label_b="SOFA"):

    y_true   = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    metrics_a = binary_metrics(y_true, y_pred_a, label=label_a)
    metrics_b = binary_metrics(y_true, y_pred_b, label=label_b)
    metrics_df = pd.concat([metrics_a, metrics_b], axis=1).T

    both_correct   = ((y_pred_a == y_true) & (y_pred_b == y_true)).sum()
    a_only_correct = ((y_pred_a == y_true) & (y_pred_b != y_true)).sum()
    b_only_correct = ((y_pred_a != y_true) & (y_pred_b == y_true)).sum()
    both_wrong     = len(y_true) - (both_correct + a_only_correct + b_only_correct)

    table = [[both_correct, a_only_correct],
             [b_only_correct, both_wrong]]

    mcnemar_res = mcnemar(table, exact=False, correction=True)

    return metrics_df, mcnemar_res


# ---------------------------------------------------------------------------
# plotting helper
# ---------------------------------------------------------------------------

def plot_accuracy_bars(metrics_df, title="Classification accuracy"):
    fig, ax = plt.subplots()
    xpos = np.arange(len(metrics_df))

    acc      = metrics_df["accuracy"].values
    err_low  = acc - metrics_df["acc_CI_low"].values
    err_up   = metrics_df["acc_CI_high"].values - acc

    ax.bar(xpos, acc, yerr=[err_low, err_up], capsize=6, color="#8ca6c8", alpha=1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(metrics_df.index)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()


def plot_outcome_heatmaps(runs,
                          yhat_thrs=np.arange(0.50, 1.01, 0.01),
                          y_thrs   =np.arange(2, 9)):
    """
    Draw three side-by-side heat-maps showing the proportion of runs
    in which the DBN fires earlier, later, or in a tie relative to a
    clinical SOFA threshold.

    Parameters
    ----------
    runs : list of tuples
        Each element must be (y_hat, ts_hat, y, ts_sofa) exactly as in your
        existing `correct_septic_runs`.
    yhat_thrs : 1-D array-like, optional
        Probability thresholds to test for the DBN output.
    y_thrs : 1-D array-like, optional
        Integer SOFA thresholds to test.

    Returns
    -------
    None
    """
    n_y   = len(y_thrs)
    n_hat = len(yhat_thrs)

    # Matrices for the three outcomes
    early  = np.full((n_y, n_hat), np.nan)
    late   = np.full_like(early, np.nan)
    tied   = np.full_like(early, np.nan)

    # ---- build the outcome grids ------------------------------------------
    for i, y_thr in enumerate(y_thrs):
        for j, h_thr in enumerate(yhat_thrs):

            e = l = t = tot = 0
            for y_hat, _ts_hat, y, _ts_sofa in runs:
                s_hat = steps_to_threshold(y_hat, h_thr)
                s_y   = steps_to_threshold(y,    y_thr)
                if s_hat is None or s_y is None:
                    continue

                tot += 1
                if   s_hat < s_y: e += 1
                elif s_hat > s_y: l += 1
                else:             t += 1

            if tot:
                early[i, j] = e / tot
                late[i, j]  = l / tot
                tied[i, j]  = t / tot
    # -----------------------------------------------------------------------

    # ---- plotting ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4),
                             sharex=True, sharey=True)

    vmin, vmax = 0.0, 1.0   # common colour scale
    titles = ("DBN fires earlier",
              "DBN fires later",
              "Tie")

    for ax, M, title in zip(axes,
                            (early, late, tied),
                            titles):
        im = ax.pcolormesh(yhat_thrs, y_thrs, M,
                           shading="auto", cmap="viridis",
                           vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("DBN probability threshold")

    axes[0].set_ylabel("SOFA threshold")
    fig.colorbar(im, ax=axes, shrink=0.8,
                 label="Proportion")
    fig.suptitle("Outcome proportions across (DBN, SOFA) threshold grid",
                 y=1.03, fontsize=14)
    plt.tight_layout()
    plt.show()
