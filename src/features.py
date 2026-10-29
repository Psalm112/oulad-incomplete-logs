"""Build one feature row per student per module run, as known at day 28.

Leakage rules enforced here:
  * studentVle is filtered to date <= CUTOFF while it is being read, before any
    aggregation happens.
  * Only assessments *submitted* on or before CUTOFF are used, and exams are dropped.
  * The unregistration date is used only to exclude students who had already left;
    it never becomes a feature.
  * final_result is used only to make the label.

Students are identified by an integer `sid`, one per (code_module,
code_presentation, id_student), so the corruption code can drop rows quickly.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CUTOFF = 28
KEY = ["code_module", "code_presentation", "id_student"]

CAT_COLS = ["age_band", "highest_education", "imd_band", "disability"]
BACKGROUND_NUM = ["num_of_prev_attempts", "studied_credits", "date_registration"]

# Day ranges for weekly click counts (inclusive). Week 4 runs to day 28, so it has 8 days.
WEEKS = {"clicks_pre": (-10_000, -1), "clicks_w1": (0, 6), "clicks_w2": (7, 13),
         "clicks_w3": (14, 20), "clicks_w4": (21, CUTOFF)}
N_TOP_TYPES = 6
# Earliest VLE date in OULAD is day -25, so a student with no activity at all gets a
# gap one day longer than anyone who was ever active.
NO_ACTIVITY_GAP = CUTOFF + 25 + 1

ASSESS_COLS = ["n_submitted", "mean_score", "n_late"]

# Ordered levels so the categoricals sort sensibly in tables and plots.
LEVELS = {
    "age_band": ["0-35", "35-55", "55<="],
    "highest_education": ["No Formal quals", "Lower Than A Level", "A Level or Equivalent",
                          "HE Qualification", "Post Graduate Qualification"],
    "imd_band": ["0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%",
                 "60-70%", "70-80%", "80-90%", "90-100%"],
    "disability": ["N", "Y"],
}


def load_students(data_dir=DATA_DIR, cutoff=CUTOFF):
    """Label, background features and exclusions. One row per student per module run."""
    info = pd.read_csv(data_dir / "studentInfo.csv", na_values="?")
    reg = pd.read_csv(data_dir / "studentRegistration.csv", na_values="?")
    df = info.merge(reg, on=KEY, how="left", validate="one_to_one")

    n_all = len(df)
    left_early = df["date_unregistration"].le(cutoff)
    not_yet_registered = df["date_registration"].gt(cutoff)
    df = df[~left_early & ~not_yet_registered].copy()

    df["at_risk"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype("int8")
    df["year"] = df["code_presentation"].str[:4].astype(int)
    for col in CAT_COLS:
        df[col] = pd.Categorical(df[col], categories=LEVELS[col], ordered=False)

    # Gender is deliberately not used (fairness). Unregistration date is dropped here so
    # it cannot leak into the features by accident.
    df = df.drop(columns=["gender", "region", "date_unregistration"])
    df = df.reset_index(drop=True)
    df.insert(0, "sid", np.arange(len(df), dtype=np.int32))
    df.attrs["exclusions"] = {
        "all_rows": n_all,
        "unregistered_by_cutoff": int(left_early.sum()),
        "registered_after_cutoff": int((not_yet_registered & ~left_early).sum()),
        "kept": len(df),
    }
    return df


def _sid_lookup(students):
    return students[KEY + ["sid"]]


def load_vle(students, data_dir=DATA_DIR, cutoff=CUTOFF, cache=True):
    """studentVle rows dated <= cutoff, with activity_type, for the kept students.

    The 430 MB file is read in chunks with only the needed columns and small dtypes,
    and each chunk is filtered to date <= cutoff before anything else is done.
    """
    cache_path = data_dir / f"studentVle_le{cutoff}.pkl"
    if cache and cache_path.exists():
        vle = pd.read_pickle(cache_path)
    else:
        dtypes = {"code_module": "category", "code_presentation": "category",
                  "id_student": "int32", "id_site": "int32", "date": "int16",
                  "sum_click": "int32"}
        chunks = []
        for chunk in pd.read_csv(data_dir / "studentVle.csv", dtype=dtypes,
                                 chunksize=2_000_000):
            chunks.append(chunk[chunk["date"] <= cutoff])
        vle = pd.concat(chunks, ignore_index=True)
        for col in ["code_module", "code_presentation"]:
            vle[col] = vle[col].astype(str)
        sites = pd.read_csv(data_dir / "vle.csv", usecols=["id_site", "activity_type"])
        vle = vle.merge(sites, on="id_site", how="left")
        if cache:
            vle.to_pickle(cache_path)

    assert vle["date"].max() <= cutoff, "VLE rows after the cutoff leaked in"
    vle = vle.merge(_sid_lookup(students), on=KEY, how="inner")
    vle["activity_type"] = vle["activity_type"].astype("category")
    return vle[["sid", "id_site", "date", "sum_click", "activity_type"]]


def load_assessments(students, data_dir=DATA_DIR, cutoff=CUTOFF):
    """Non-exam assessment submissions made on or before cutoff, for the kept students."""
    meta = pd.read_csv(data_dir / "assessments.csv", na_values="?")
    sub = pd.read_csv(data_dir / "studentAssessment.csv", na_values="?")
    sub = sub.merge(meta, on="id_assessment", how="left")
    sub = sub[(sub["assessment_type"] != "Exam") & (sub["date_submitted"] <= cutoff)]
    sub = sub.merge(_sid_lookup(students), on=["code_module", "code_presentation", "id_student"],
                    how="inner")
    sub["late"] = (sub["date_submitted"] > sub["date"]).astype("int8")
    assert sub["date_submitted"].max() <= cutoff
    return sub[["sid", "id_assessment", "date_submitted", "score", "late"]]


def top_activity_types(vle, sids, n=N_TOP_TYPES):
    """Most-clicked activity types among the given students (pass training students only)."""
    clicks = vle[vle["sid"].isin(sids)].groupby("activity_type", observed=True)["sum_click"].sum()
    return list(clicks.sort_values(ascending=False).index[:n])


def activity_features(vle, sids, top_types):
    """Click features per student. Students with no rows get zeros (the 'naive' reading)."""
    g = vle.groupby("sid")
    out = pd.DataFrame(index=pd.Index(sids, name="sid"))
    out["total_clicks"] = g["sum_click"].sum()
    for name, (lo, hi) in WEEKS.items():
        in_week = vle["date"].between(lo, hi)
        out[name] = vle[in_week].groupby("sid")["sum_click"].sum()
    out["active_days"] = g["date"].nunique()
    out["n_resources"] = g["id_site"].nunique()
    by_type = (vle[vle["activity_type"].isin(top_types)]
               .pivot_table(index="sid", columns="activity_type", values="sum_click",
                            aggfunc="sum", observed=True))
    for t in top_types:
        out[f"clicks_{t}"] = by_type[t] if t in by_type else np.nan
    out = out.fillna(0)
    out["days_since_last"] = (CUTOFF - g["date"].max()).reindex(out.index).fillna(NO_ACTIVITY_GAP)
    return out.astype("float32")


def assessment_features(sub, sids):
    """Early assessment features. mean_score is 0 when nothing was submitted; n_submitted
    carries that information."""
    g = sub.groupby("sid")
    out = pd.DataFrame(index=pd.Index(sids, name="sid"))
    out["n_submitted"] = g.size()
    out["mean_score"] = g["score"].mean()
    out["n_late"] = g["late"].sum()
    return out.fillna(0).astype("float32")


def activity_cols(top_types):
    return (["total_clicks", *WEEKS, "active_days", "n_resources"]
            + [f"clicks_{t}" for t in top_types] + ["days_since_last"])


def build_features(students, vle, sub, top_types):
    """Full feature table indexed by sid: background + activity + assessment."""
    sids = students["sid"].to_numpy()
    background = students.set_index("sid")[CAT_COLS + BACKGROUND_NUM]
    X = background.join(activity_features(vle, sids, top_types)).join(assessment_features(sub, sids))
    return X


def load_all(data_dir=DATA_DIR, cutoff=CUTOFF):
    """Convenience loader: students, filtered logs, assessments, and the top activity types
    (chosen from 2013 students only)."""
    students = load_students(data_dir, cutoff)
    vle = load_vle(students, data_dir, cutoff)
    sub = load_assessments(students, data_dir, cutoff)
    train_sids = students.loc[students["year"] == 2013, "sid"]
    top_types = top_activity_types(vle, train_sids)
    return students, vle, sub, top_types
