def _prep_targets(df, col, filtered=True):
    y_true = df[col].dropna()
    y_pred = df.loc[y_true.index, "prediction"]

    if filtered:
        mask = y_true != 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        labels = [-1, 1]
    else:
        labels = [-1, 0, 1]

    return y_true, y_pred, labels