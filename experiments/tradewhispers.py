# tradewhispers.py
"""
Evaluate TradeWhispers predictions against FX forward returns.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def map_predictions(df, col="Prediction"):
    """
    Map Portuguese predictions to: Increase, Decrease, Stable.
    """
    mapping = {
        "Em baixa": "Increase",
        "Em alta": "Decrease",
        "Manter": "Stable",
        -1: "Increase",
        1: "Decrease",
        0: "Stable"
    }
    df[col] = df[col].map(mapping)
    return df

def label_ground_truth(series, threshold_pips):
    """
    Label ground truth from forward returns using pip threshold.
    abs(x) <= threshold -> Stable, x > threshold -> Increase, x < -threshold -> Decrease.
    """
    return series.map(lambda x: (
        np.nan if pd.isna(x) else
        "Stable" if abs(x) <= threshold_pips else
        "Increase" if x > threshold_pips else
        "Decrease"
    ))

def evaluate(stage_df, results_df, horizons, threshold_pips, mode):
    """
    Compare predictions to ground truth for each horizon.

    - Accuracy from matches between y_true and y_pred.
    - PnL = sum of forward returns where prediction was Increase,
      minus sum where prediction was Decrease. Stable = no trade.
    - Prints confusion matrix, classification report, accuracy, and PnL.
    - Returns summary DataFrame.
    """
    # Ensure one stage row per Timestamp to avoid duplication after merge
    stage_unique = stage_df.drop_duplicates(subset=["Timestamp"]).copy()

    # Keep only rows in stage that exist in results to make counts reflect results file rows
    merged = pd.merge(stage_unique, results_df[["Timestamp", "Prediction"]], on="Timestamp", how="inner")

    # Use row counts (not unique timestamps)
    n_results = len(results_df)
    n_merged = len(merged)
    if n_merged != n_results:
        print(f"[Info] Results file rows: {n_results}; matched rows after merge: {n_merged}.")

    summary = []

    for h in horizons:
        col = f"Forward Return t+{h}"
        if col not in merged.columns:
            continue

        # Evaluate only on matched examples that also have a realized forward return AND a non-null prediction
        df_h = merged.dropna(subset=[col, "Prediction"]).copy()
        df_h[f"{col}_pips"] = df_h[col] / 0.0001

        y_true = label_ground_truth(df_h[f"{col}_pips"], threshold_pips)
        y_pred = df_h["Prediction"]

        # Drop any rows where labeling produced NaN
        valid_mask = y_true.notna() & y_pred.notna()
        y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]
        df_h = df_h.loc[valid_mask]

        if mode == "filtered":
            trade_mask = y_pred != "Stable"
            y_true, y_pred = y_true[trade_mask], y_pred[trade_mask]
            labels = ["Decrease", "Increase"]
            n_traded = int(trade_mask.sum())
        else:
            labels = ["Decrease", "Stable", "Increase"]
            n_traded = int((df_h["Prediction"] != "Stable").sum())

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cr = classification_report(y_true, y_pred, labels=labels, zero_division=0)

        # Align returns to the (possibly filtered) y_pred index
        pnl = ((y_pred == "Increase") * df_h.loc[y_pred.index, col] -
               (y_pred == "Decrease") * df_h.loc[y_pred.index, col]).sum()
        acc = (y_true == y_pred).mean()

        n_eval = len(y_true)  # number of evaluated rows for this horizon

        summary.append({
            "horizon": h,
            "n_results": n_results,   # total rows in results file
            "n_matched": n_merged,    # rows that matched by Timestamp
            "n_evaluated": n_eval,    # rows with realized forward returns at this horizon
            "n_traded": n_traded,     # predictions implying a trade (context)
            "accuracy": acc,
            "cum_pnl": pnl
        })

        print(f"\n=== Horizon t+{h} ({mode}) ===")
        print(f"Results rows: {n_results} | Matched: {n_merged} | Evaluated (has {col}): {n_eval} | Traded preds: {n_traded}")
        print("Confusion Matrix:\n", pd.DataFrame(cm, index=labels, columns=labels))
        print("\nClassification Report:\n", cr)
        print("Accuracy:", round(acc, 4))
        print("Cumulative PnL:", pnl)

    return pd.DataFrame(summary)

def simulate_trading_account(stage_df, results_df, horizons, threshold_pips, mode, starting_balance=1000):
    """
    Simulate a trading account with compounding balance.

    - Increase → balance *= (1 + return).
    - Decrease → balance *= (1 - return).
    - Stable → no change.
    Returns a DataFrame of final balances per horizon.
    """
    merged = pd.merge(stage_df, results_df[["Timestamp","Prediction"]], on="Timestamp", how="inner")
    balances = []

    for h in horizons:
        col = f"Forward Return t+{h}"
        if col not in merged.columns:
            continue

        # Exclude any rows with NA forward return or NA prediction
        df_h = merged.dropna(subset=[col, "Prediction"]).copy()
        df_h[f"{col}_pips"] = df_h[col] / 0.0001
        y_pred = df_h["Prediction"]

        if mode == "filtered":
            mask = y_pred != "Stable"
            df_h, y_pred = df_h[mask], y_pred[mask]

        balance = starting_balance
        for ret, pred in zip(df_h[col], y_pred):
            if pred == "Increase":
                balance *= (1 + ret)
            elif pred == "Decrease":
                balance *= (1 - ret)
            # Stable → no change
        balances.append({"horizon": h, "final_balance": balance})

    balances_df = pd.DataFrame(balances)
    print(f"\n==== Trading Account Simulation ({mode}) ====")
    display(balances_df)
    return balances_df