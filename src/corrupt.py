"""Controlled damage to the raw records, applied before features are rebuilt.

A  drop_random_rows   - each studentVle row is lost independently with probability p.
B  fail_students      - for a fraction q of students, every studentVle row is lost.
C  drop_scores        - for a fraction q of students, every early assessment record is lost.

Each function returns the damaged table and, for B and C, the boolean mask of affected
students (aligned with `sids`). The mask is what a system would know only if it records
its own logging failures; the "indicators" strategy uses it, the "naive" one does not.
"""
import numpy as np
import pandas as pd

from .features import ASSESS_COLS, activity_cols


def drop_random_rows(vle, p, rng):
    """Damage A: lose a fraction p of log rows at random."""
    if p <= 0:
        return vle
    return vle[rng.random(len(vle)) >= p]


def drop_rows_per_student(vle, sids, rates, rng):
    """Damage A for training: each student gets their own loss rate, drawn from `rates`."""
    student_rate = pd.Series(rng.choice(rates, size=len(sids)), index=sids)
    row_rate = student_rate.reindex(vle["sid"]).to_numpy()
    return vle[rng.random(len(vle)) >= row_rate]


def pick_students(sids, q, rng, among=None):
    """Boolean mask over `sids` selecting a fraction q of them (optionally only from `among`)."""
    sids = np.asarray(sids)
    pool = sids if among is None else np.intersect1d(sids, among)
    chosen = rng.choice(pool, size=int(round(q * len(pool))), replace=False)
    return np.isin(sids, chosen)


def fail_students(vle, sids, q, rng, among=None):
    """Damage B: whole-student logging failure for a fraction q of students."""
    failed = pick_students(sids, q, rng, among)
    return vle[~vle["sid"].isin(np.asarray(sids)[failed])], failed


def drop_scores(sub, sids, q, rng, among=None):
    """Damage C: all early assessment records missing for a fraction q of students."""
    lost = pick_students(sids, q, rng, among)
    return sub[~sub["sid"].isin(np.asarray(sids)[lost])], lost


def add_indicators(X, top_types, log_failed=None, scores_missing=None):
    """Missing-data indicator handling: blank the affected values and add a flag.

    Values become NaN rather than 0, so the gradient-boosting model can route them
    separately and logistic regression imputes them; the flag column says why.
    """
    X = X.copy()
    log_failed = np.zeros(len(X), bool) if log_failed is None else np.asarray(log_failed)
    scores_missing = np.zeros(len(X), bool) if scores_missing is None else np.asarray(scores_missing)
    X.loc[log_failed, activity_cols(top_types)] = np.nan
    X.loc[scores_missing, ASSESS_COLS] = np.nan
    X["log_failed"] = log_failed.astype("float32")
    X["scores_missing"] = scores_missing.astype("float32")
    return X
