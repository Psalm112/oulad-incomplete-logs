"""Models, metrics and bootstrap confidence intervals."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MODELS = ["baseline", "logreg", "hgb"]
MODEL_LABELS = {"baseline": "Majority class", "logreg": "Logistic regression",
                "hgb": "Gradient boosting"}
THRESHOLD = 0.5


def make_model(name, X, seed=0):
    """Unfitted model for feature frame X (column types decide the preprocessing)."""
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    nums = [c for c in X.columns if c not in cats]
    if name == "baseline":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        # Click counts are heavily right-skewed, so they are log1p-transformed before scaling.
        counts = [c for c in nums if c.startswith("clicks_") or c == "total_clicks"]
        others = [c for c in nums if c not in counts]
        numeric = lambda *steps: Pipeline([("impute", SimpleImputer(strategy="median")),
                                           *steps, ("scale", StandardScaler())])
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
            ("counts", numeric(("log", FunctionTransformer(np.log1p))), counts),
            ("num", numeric(), others),
        ])
        clf = LogisticRegression(class_weight="balanced", max_iter=5000)
        return Pipeline([("pre", pre), ("clf", clf)])
    if name == "hgb":
        # Categoricals are passed as pandas categories; NaN is handled natively.
        return HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=seed)
    raise ValueError(name)


def fit_predict(name, X_train, y_train, X_test, seed=0):
    model = make_model(name, X_train, seed).fit(X_train, y_train)
    return model, model.predict_proba(X_test[X_train.columns])[:, 1]


def metrics(y, p, threshold=THRESHOLD):
    """Headline metrics for the at-risk class (label 1)."""
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    single_class = len(np.unique(p)) == 1
    return {
        "auc": 0.5 if single_class else roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "false_alarm_rate": fp / (fp + tn) if fp + tn else np.nan,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def false_alarm_rate(y, p, mask, threshold=THRESHOLD):
    """Share of not-at-risk students in `mask` who are flagged at risk."""
    y, p, mask = np.asarray(y), np.asarray(p), np.asarray(mask, bool)
    healthy = mask & (y == 0)
    return (p[healthy] >= threshold).mean() if healthy.any() else np.nan


def bootstrap_auc(y, p, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for ROC-AUC, resampling test rows with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        aucs[b] = roc_auc_score(y[idx], p[idx]) if y[idx].min() != y[idx].max() else np.nan
    lo, hi = np.nanpercentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def confusion_table(y, p, threshold=THRESHOLD):
    cm = confusion_matrix(np.asarray(y), (np.asarray(p) >= threshold).astype(int), labels=[0, 1])
    return pd.DataFrame(cm, index=["true: not at risk", "true: at risk"],
                        columns=["pred: not at risk", "pred: at risk"])
