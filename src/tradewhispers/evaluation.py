import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from .utils import _prep_targets

def classification_reports(df, filtered=True):
    frames = {}
    for col in [c for c in df.columns if c.startswith("t+")]:
        y_true, y_pred, labels = _prep_targets(df, col, filtered)
        cr = classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        )
        frames[col] = pd.DataFrame(cr).transpose()
    return pd.concat(frames)


def confusion_matrices(df, filtered=True):
    frames = {}
    for col in [c for c in df.columns if c.startswith("t+")]:
        y_true, y_pred, labels = _prep_targets(df, col, filtered)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        frames[col] = pd.DataFrame(cm, index=labels, columns=labels)
    return pd.concat(frames)