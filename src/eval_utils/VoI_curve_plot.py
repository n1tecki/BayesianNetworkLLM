import random
import matplotlib.pyplot as plt
import pandas as pd

def plot_random_stay_trajectories(sofa_df, df_test, voi_pred_dict, test_ids, hadm=None, VOI_THR=0.5, SOFA_THR=2):
    """
    Pick one random stay (hadm_id) from test_ids, then plot:
      - Left:  SOFA-total over time
      - Right: VoI probability (p_post) over time

    Parameters
    ----------
    sofa_df : pandas.DataFrame
        Indexed by (hadm_id, timestamp), must contain a "sofa_total" column.
    df_test : pandas.DataFrame
        Must contain rows for each (hadm_id, timestamp) that correspond to the VoI predictions.
    voi_pred_dict : dict
        Maps str(hadm_id) -> list-of-dicts, each dict having keys "step", "p_post", etc.
    test_ids : Index (or list-like) of hadm_id values that are in df_test.
    """
    if not hadm:
        hadm = random.choice(list(test_ids))
    print(f"Selected random stay: hadm_id = {hadm}")

    # 2. Extract VoI timeline for this stay, sorted by step ≥ 0
    raw_records = voi_pred_dict[str(hadm)]
    # Keep only records with step >= 0, then sort by "step"
    steps = sorted((rec for rec in raw_records if rec["step"] >= 0),
                   key=lambda rec: rec["step"])
    # y_voi is the list of posterior probabilities at each recorded step
    y_voi = [rec["p_post"] for rec in steps]

    # 3. Get the matching timestamps from df_test for this stay
    voi_rows = df_test.loc[hadm]
    if isinstance(voi_rows, pd.Series):
        voi_rows = voi_rows.to_frame().T
    voi_rows = voi_rows.sort_values("timestamp")
    ts_voi = pd.to_datetime(voi_rows["timestamp"]).tolist()

    # Align lengths in case JSON vs. df_test differ
    L_voi = min(len(y_voi), len(ts_voi))
    y_voi, ts_voi = y_voi[:L_voi], ts_voi[:L_voi]

    # 4. Prepend a “default” probability = 0.5 at step -1,
    #    placing it just one second before the first real lab time.
    if ts_voi:
        ts_prior = ts_voi[0] - pd.Timedelta(seconds=1)
        y_voi = [0.5] + y_voi
        ts_voi = [ts_prior] + ts_voi

    # 5. Extract the SOFA timeline for this stay
    sofa_rows = sofa_df.loc[hadm]
    if isinstance(sofa_rows, pd.Series):
        sofa_rows = sofa_rows.to_frame().T
    sofa_rows = sofa_rows.sort_index(level="timestamp")
    y_sofa  = sofa_rows["sofa_total"].tolist()
    ts_sofa = pd.to_datetime(sofa_rows.index).tolist()

    # Align lengths for SOFA (just in case)
    L_sofa = min(len(y_sofa), len(ts_sofa))
    y_sofa, ts_sofa = y_sofa[:L_sofa], ts_sofa[:L_sofa]

    # 6. Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)

    # 6a. Left subplot: SOFA over time with a horizontal line at SOFA_THR
    axes[0].plot(ts_sofa, y_sofa, marker="o", linestyle="-", label="SOFA total")
    axes[0].axhline(
        SOFA_THR,
        color="red",
        linestyle="--",
        label=f"SOFA threshold = {SOFA_THR}"
    )
    axes[0].set_title(f"SOFA-total trajectory\n(hadm_id={hadm})")
    axes[0].set_xlabel("Timestamp")
    axes[0].set_ylabel("SOFA total")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()

    # 6b. Right subplot: VoI/DBN probability over time with two points:
    #     (1) step -1 at y=0.5, (2…) actual p_post values
    axes[1].plot(ts_voi, y_voi, marker="o", linestyle="-", color="#1f77b4",
                 label="VoI p(sepsis)")
    axes[1].axhline(
        VOI_THR,
        color="red",
        linestyle="--",
        label=f"VoI threshold = {VOI_THR}"
    )
    axes[1].set_title(f"VoI/DBN p(sepsis) trajectory\n(hadm_id={hadm})")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("Probability")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()

    plt.tight_layout()
    plt.show()