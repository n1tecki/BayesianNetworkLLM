"""
cpds_plot.py
Plot selected TabularCPD objects as heat-maps (incl. numbers).
Works with pgmpy <1.0 and ≥1.0 and any data-dimensionality.
"""

import math
import textwrap
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def _tuple_to_str(var):
    """('creatinin', 1) -> 'creatinin_1'   |   'sepsis' -> 'sepsis'"""
    return f"{var[0]}_{var[1]}" if isinstance(var, tuple) else str(var)


def _get_evidence_vars(cpd):
    """
    pgmpy ≤0.1.x: TabularCPD.evidence (property)
    pgmpy ≥1.0:   TabularCPD.get_evidence() (method)
    """
    if hasattr(cpd, "evidence"):
        return list(cpd.evidence or [])
    if hasattr(cpd, "get_evidence"):
        return list(cpd.get_evidence() or [])
    return []


def _make_state_labels(cpd, evidence_vars, wrap_len=12):
    """
    Build y_labels (target states) and x_labels (cartesian product of evidence states).
    """
    # --- rows: target states
    if getattr(cpd, "state_names", None) and cpd.variable in cpd.state_names:
        y_labels = [str(s) for s in cpd.state_names[cpd.variable]]
    else:
        idx = cpd.variables.index(cpd.variable)
        y_labels = list(map(str, range(cpd.cardinality[idx])))

    # --- columns: evidence combos
    if not evidence_vars:
        return y_labels, [""]  # single prior column

    lists = []
    for ev in evidence_vars:
        if getattr(cpd, "state_names", None) and ev in cpd.state_names:
            lists.append([str(s) for s in cpd.state_names[ev]])
        else:
            idx = cpd.variables.index(ev)
            lists.append(list(map(str, range(cpd.cardinality[idx]))))

    combos = list(product(*lists))
    col_labels = []
    for combo in combos:
        parts = [f"{_tuple_to_str(ev)}={st}" for ev, st in zip(evidence_vars, combo)]
        label = "|".join(parts)
        col_labels.append("\n".join(textwrap.wrap(label, width=wrap_len)))
    return y_labels, col_labels


def plot_all_cpds_heatmap(all_cpds, *,
                          indices=None,
                          variables=None,
                          n_cols=3,
                          wrap_len=12,
                          cmap="viridis",
                          annot_fmt=".2f",
                          figsize_per_subplot=(4, 4)):
    """
    Parameters
    ----------
    all_cpds : list[TabularCPD]
        The full CPD list, e.g. model.get_cpds().
    indices : list[int], optional
        If given, only plot CPDs at these positions.
    variables : list[tuple|str], optional
        If given, only plot CPDs whose variable matches any selector.
        A selector can be the exact tuple (e.g. ('creatinin', 1)) or
        just the name string (e.g. 'creatinin') to include all slices.
    n_cols : int
        Number of heatmaps per row.
    wrap_len : int
        Wrap long x-tick labels every N chars.
    cmap : str
        Matplotlib colormap name.
    annot_fmt : str
        Numeric format for cell annotations.
    figsize_per_subplot : (w, h)
        Inches per small heatmap.
    """
    # --- select subset ------------------------------------------------------
    cpds = list(all_cpds)
    if indices is not None:
        cpds = [cpds[i] for i in indices]
    elif variables is not None:
        def match(var, sel):
            if isinstance(var, tuple):
                return sel == var or sel == var[0]
            return sel == var
        cpds = [
            cpd for cpd in cpds
            if any(match(cpd.variable, sel) for sel in variables)
        ]

    n = len(cpds)
    if n == 0:
        raise ValueError("No CPDs match the given indices/variables.")

    # --- prepare subplot grid ----------------------------------------------
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_subplot[0] * n_cols,
                                      figsize_per_subplot[1] * n_rows),
                             squeeze=False)

    vmin, vmax = 0.0, 1.0  # shared color scale

    for i, cpd in enumerate(cpds):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]

        # --- data → 2D matrix --------------------------------------------
        data = np.asarray(cpd.values, dtype=float)
        if data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        elif data.ndim == 1:
            data = data.reshape(data.shape[0], 1)

        # --- labels ------------------------------------------------------
        evidence_vars = _get_evidence_vars(cpd)
        y_labels, x_labels = _make_state_labels(cpd, evidence_vars, wrap_len)

        # --- draw heatmap -----------------------------------------------
        sns.heatmap(
            data, ax=ax,
            cmap=cmap, vmin=vmin, vmax=vmax,
            annot=True, fmt=annot_fmt,
            cbar=False,
            xticklabels=x_labels,
            yticklabels=y_labels,
            square=True
        )

        # --- title ------------------------------------------------------
        tgt = _tuple_to_str(cpd.variable)
        if evidence_vars:
            ev = ", ".join(_tuple_to_str(e) for e in evidence_vars)
            ax.set_title(f"P({tgt} | {ev})", fontsize=10)
        else:
            ax.set_title(f"P({tgt})", fontsize=10)

        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    # turn off any unused axes
    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")

    # --- shared colorbar ---------------------------------------------------
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Probability")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


# -------------------------- example usage -----------------------------------
if __name__ == "__main__":
    # from your_model import model
    # all_cpds = model.get_cpds()
    #
    # # 1) by index:
    # plot_all_cpds_heatmap(all_cpds, indices=[0, 3, 7], n_cols=3)
    #
    # # 2) by variable (both slices if just name):
    # plot_all_cpds_heatmap(all_cpds,
    #                       variables=[('creatinin',1), 'sepsis'],
    #                       n_cols=2)
    pass
