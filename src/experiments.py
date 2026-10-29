"""All experiments, end to end. Run from the repo root with:

    python -m src.experiments

Writes tables and figures to results/.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import corrupt
from .evaluate import (MODEL_LABELS, MODELS, bootstrap_auc, confusion_table, false_alarm_rate,
                       fit_predict, make_model, metrics)
from .features import DATA_DIR, KEY, build_features, load_all

RESULTS = Path(__file__).resolve().parents[1] / "results"

LEVELS = {"A": [0.0, 0.1, 0.2, 0.3, 0.5], "B": [0.0, 0.1, 0.2, 0.3], "C": [0.0, 0.1, 0.2, 0.3]}
DAMAGE_LABELS = {"A": "A. Random record loss (% of log rows)",
                 "B": "B. Whole-student logging failure (% of students)",
                 "C": "C. Missing early assessment scores (% of students)"}
# Random row loss can't be detected per student, so there is nothing to flag for A.
STRATEGIES = {"A": ["naive", "train_damage"],
              "B": ["naive", "indicators", "train_damage", "indicators+train_damage"],
              "C": ["naive", "indicators", "train_damage", "indicators+train_damage"]}
STRATEGY_LABELS = {"naive": "Naive (missing = zero)", "indicators": "Missing-data indicators",
                   "train_damage": "Train with damage",
                   "indicators+train_damage": "Indicators + train with damage"}
# Damage applied to the 2013 training logs for the "train with damage" strategies.
TRAIN_ROW_LOSS_RATES = [0.0, 0.1, 0.2, 0.3, 0.5]   # A: each student gets one of these
TRAIN_STUDENT_FRACTION = 0.15                       # B and C
N_SEEDS = 5
CI_LEVELS = (0.0, 0.3)
EVAL_MODELS = ["logreg", "hgb"]


class Data:
    """Clean inputs plus the time split (train 2013, test 2014)."""

    def __init__(self):
        self.students, self.vle, self.sub, self.top_types = load_all()
        s = self.students
        self.X = build_features(s, self.vle, self.sub, self.top_types)
        self.y = s.set_index("sid")["at_risk"]
        self.train = s[s["year"] == 2013]
        self.test = s[s["year"] == 2014]
        self.train_sids = self.train["sid"].to_numpy()
        self.test_sids = self.test["sid"].to_numpy()
        self.vle_train = self.vle[self.vle["sid"].isin(self.train_sids)]
        self.vle_test = self.vle[self.vle["sid"].isin(self.test_sids)]
        self.sub_train = self.sub[self.sub["sid"].isin(self.train_sids)]
        self.sub_test = self.sub[self.sub["sid"].isin(self.test_sids)]


def _row(split, model, y, p, n_boot, seed=0):
    m = metrics(y, p)
    m["auc_lo"], m["auc_hi"] = bootstrap_auc(y, p, n_boot, seed) if model != "baseline" else (0.5, 0.5)
    return {"split": split, "model": model, **m}


def clean_results(d, n_boot=1000):
    """Day 2: baseline, logistic regression and gradient boosting on clean data.
    Time split (main) and a stratified random 80/20 split (comparison)."""
    rows, confusions = [], {}
    Xtr, ytr = d.X.loc[d.train_sids], d.y.loc[d.train_sids]
    Xte, yte = d.X.loc[d.test_sids], d.y.loc[d.test_sids]
    for name in MODELS:
        _, p = fit_predict(name, Xtr, ytr, Xte)
        rows.append(_row("time (2013 -> 2014)", name, yte, p, n_boot))
        confusions[name] = confusion_table(yte, p)

    idx_tr, idx_te = train_test_split(d.X.index, test_size=0.2, stratify=d.y, random_state=0)
    for name in MODELS:
        _, p = fit_predict(name, d.X.loc[idx_tr], d.y.loc[idx_tr], d.X.loc[idx_te])
        rows.append(_row("random 80/20", name, d.y.loc[idx_te], p, n_boot))
    return pd.DataFrame(rows), confusions


def leaky_demo(d):
    """What leakage looks like: the same model, but activity features computed over the
    whole module run instead of days <= 28. Not used anywhere else."""
    dtypes = {"code_module": "category", "code_presentation": "category", "id_student": "int32",
              "id_site": "int32", "date": "int16", "sum_click": "int32"}
    full = pd.read_csv(DATA_DIR / "studentVle.csv", dtype=dtypes)
    for col in ["code_module", "code_presentation"]:
        full[col] = full[col].astype(str)
    full = full.merge(d.students[KEY + ["sid"]], on=KEY, how="inner")
    full = full.merge(pd.read_csv(DATA_DIR / "vle.csv", usecols=["id_site", "activity_type"]),
                      on="id_site", how="left")
    full["activity_type"] = full["activity_type"].astype("category")
    X_leaky = build_features(d.students, full, d.sub, d.top_types)
    # build_features measures days_since_last from day 28; over the whole run that is
    # negative and itself encodes when the student stopped, which is the point here.
    _, p = fit_predict("hgb", X_leaky.loc[d.train_sids], d.y.loc[d.train_sids],
                       X_leaky.loc[d.test_sids])
    return metrics(d.y.loc[d.test_sids], p)["auc"]


def _damage(d, damage, vle, sub, sids, level, rng, students):
    """Apply one damage type to the given logs and rebuild features. Returns (X, mask)."""
    mask = np.zeros(len(sids), bool)
    if damage == "A":
        vle = corrupt.drop_random_rows(vle, level, rng)
    elif damage == "B":
        vle, mask = corrupt.fail_students(vle, sids, level, rng)
    elif damage == "C":
        sub, mask = corrupt.drop_scores(sub, sids, level, rng)
    return build_features(students, vle, sub, d.top_types), mask


def _handle(d, damage, strategy, X, mask):
    """Turn rebuilt features into what a strategy feeds the model."""
    if not strategy.startswith("indicators"):
        return X
    return corrupt.add_indicators(X, d.top_types,
                                  log_failed=mask if damage == "B" else None,
                                  scores_missing=mask if damage == "C" else None)


def training_set(d, damage, strategy, seed=1000):
    rng = np.random.default_rng(seed)
    if "train_damage" not in strategy:
        X, mask = d.X.loc[d.train_sids], np.zeros(len(d.train_sids), bool)
    elif damage == "A":
        vle = corrupt.drop_rows_per_student(d.vle_train, d.train_sids, TRAIN_ROW_LOSS_RATES, rng)
        X, mask = build_features(d.train, vle, d.sub_train, d.top_types), np.zeros(len(d.train_sids), bool)
    else:
        X, mask = _damage(d, damage, d.vle_train, d.sub_train, d.train_sids,
                          TRAIN_STUDENT_FRACTION, rng, d.train)
    return _handle(d, damage, strategy, X, mask)


def run_damage(d, damage, n_seeds=N_SEEDS, n_boot=1000, verbose=True):
    """Train on 2013 (clean or damaged per strategy), damage the 2014 test records at each
    level and seed, and score every model x strategy on identical damaged data."""
    ytr = d.y.loc[d.train_sids]
    yte = d.y.loc[d.test_sids].to_numpy()
    models = {}
    for strategy in STRATEGIES[damage]:
        Xtr = training_set(d, damage, strategy)
        for name in EVAL_MODELS:
            models[strategy, name] = (Xtr.columns, make_model(name, Xtr).fit(Xtr, ytr))

    rows = []
    for level in LEVELS[damage]:
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            X, mask = _damage(d, damage, d.vle_test, d.sub_test, d.test_sids, level, rng, d.test)
            for strategy in STRATEGIES[damage]:
                Xs = _handle(d, damage, strategy, X, mask)
                for name in EVAL_MODELS:
                    cols, model = models[strategy, name]
                    p = model.predict_proba(Xs[cols])[:, 1]
                    m = metrics(yte, p)
                    m["false_alarm_affected"] = false_alarm_rate(yte, p, mask)
                    m["false_alarm_unaffected"] = false_alarm_rate(yte, p, ~mask)
                    m["auc_lo"] = m["auc_hi"] = np.nan
                    if seed == 0 and level in CI_LEVELS:
                        m["auc_lo"], m["auc_hi"] = bootstrap_auc(yte, p, n_boot)
                    rows.append({"damage": damage, "level": level, "seed": seed,
                                 "strategy": strategy, "model": name, **m})
            if verbose:
                print(f"  {damage} level={level:.0%} seed={seed} done", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- outputs

# Validated categorical slots 1-4 (blue, orange, aqua, yellow) on a light surface.
STRATEGY_COLORS = {"naive": "#2a78d6", "indicators": "#eb6834", "train_damage": "#1baf7a",
                   "indicators+train_damage": "#eda100"}
MODEL_STYLES = {"hgb": dict(ls="-", marker="o"), "logreg": dict(ls="--", marker="s")}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)


def _curve_figure(df, metric, ylabel, damages, path, ylim=None, ref=None):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = (df.groupby(["damage", "strategy", "model", "level"])[metric]
               .agg(["mean", "min", "max"]).reset_index())
    fig, axes = plt.subplots(1, len(damages), figsize=(5.2 * len(damages), 4.2),
                             sharey=ylim is not None, facecolor=SURFACE)
    for ax, damage in zip(np.atleast_1d(axes), damages):
        sub = summary[summary["damage"] == damage]
        for strategy in STRATEGIES[damage]:
            for model in EVAL_MODELS:
                s = sub[(sub["strategy"] == strategy) & (sub["model"] == model)]
                x = s["level"] * 100
                ax.fill_between(x, s["min"], s["max"], color=STRATEGY_COLORS[strategy],
                                alpha=0.15, lw=0)
                ax.plot(x, s["mean"], color=STRATEGY_COLORS[strategy], lw=2, ms=5,
                        label=f"{STRATEGY_LABELS[strategy]} - {MODEL_LABELS[model]}",
                        **MODEL_STYLES[model])
        _style(ax, DAMAGE_LABELS[damage], "% missing", ylabel)
        ax.set_xticks(sorted(sub["level"].unique() * 100))
        if ylim:
            ax.set_ylim(*ylim)
        if ref:
            ax.axhline(ref[0], color=INK_2, lw=1, ls=":")
            ax.annotate(ref[1], (0.01, ref[0]), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK_2)
    handles, labels = np.atleast_1d(axes)[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def make_figures(damage_df, clean):
    _curve_figure(damage_df, "auc", "ROC-AUC (2014 test set)", ["A", "B", "C"],
                  RESULTS / "auc_vs_missing.png", ylim=(0.67, 0.76))
    clean_fa = clean.set_index(["split", "model"]).loc[("time (2013 -> 2014)", "hgb"), "false_alarm_rate"]
    _curve_figure(damage_df[damage_df["damage"].isin(["B", "C"])], "false_alarm_affected",
                  "False-alarm rate among affected\nnot-at-risk students", ["B", "C"],
                  RESULTS / "false_alarms.png", ylim=(0, 1),
                  ref=(clean_fa, f"clean, GB: {clean_fa:.0%}"))


def write_outputs(clean, confusions, damage_df, leaky_auc=None):
    RESULTS.mkdir(exist_ok=True)
    clean.to_csv(RESULTS / "clean_results.csv", index=False)
    with open(RESULTS / "confusion_matrices.md", "w") as f:
        for name, cm in confusions.items():
            f.write(f"### {MODEL_LABELS[name]} (time split, threshold 0.5)\n\n{to_markdown(cm.reset_index(names=''))}\n\n")
    damage_df.to_csv(RESULTS / "damage_results.csv", index=False)
    if leaky_auc is not None:
        (RESULTS / "leaky_demo.txt").write_text(f"HGB AUC with whole-course activity (leaky): {leaky_auc:.3f}\n")

    make_figures(damage_df, clean)
    table = damage_table(damage_df)
    table.to_csv(RESULTS / "auc_table.csv", index=False)
    (RESULTS / "auc_table.md").write_text(to_markdown(table))
    return table


def to_markdown(df):
    """Minimal GitHub table (avoids a tabulate dependency)."""
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def damage_table(df):
    """AUC with 95% CI at 0% and 30% damage, plus false alarms (seed 0, threshold 0.5)."""
    rows = []
    s0 = df[df["seed"] == 0]
    for (damage, strategy, model), g in s0.groupby(["damage", "strategy", "model"], sort=False):
        g = g.set_index("level")
        fmt = lambda lv: f"{g.loc[lv, 'auc']:.3f} [{g.loc[lv, 'auc_lo']:.3f}, {g.loc[lv, 'auc_hi']:.3f}]"
        rows.append({
            "damage": damage, "strategy": STRATEGY_LABELS[strategy], "model": MODEL_LABELS[model],
            "AUC at 0%": fmt(0.0), "AUC at 30%": fmt(0.3),
            "false alarms at 0%": f"{g.loc[0.0, 'false_alarm_rate']:.1%}",
            "false alarms at 30%": f"{g.loc[0.3, 'false_alarm_rate']:.1%}",
            "false alarms, affected students at 30%": (
                "-" if damage == "A" else f"{g.loc[0.3, 'false_alarm_affected']:.1%}"),
        })
    return pd.DataFrame(rows)


def main():
    d = Data()
    print("students:", d.students.attrs["exclusions"])
    clean, confusions = clean_results(d)
    print(clean[["split", "model", "auc", "auc_lo", "auc_hi", "pr_auc", "recall", "f1"]].round(3))
    print("leaky demo AUC:", round(leaky := leaky_demo(d), 3))
    damage_df = pd.concat([run_damage(d, damage) for damage in ["A", "B", "C"]], ignore_index=True)
    print(write_outputs(clean, confusions, damage_df, leaky).to_string())


if __name__ == "__main__":
    main()
