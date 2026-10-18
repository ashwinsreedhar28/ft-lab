"""Surface-cue leakage probe: can a bag-of-tokens linear model tell bug PRs from clean PRs from the
DIFF TEXT ALONE? If yes, a model trained on this bench can score without reading the code, and any
GRPO gain on mutants is suspect until the real-positive number agrees.

    python -m review.analysis.leakage [--tasks review/bench/public_tasks.jsonl] [--by source]

Reports cross-validated AUROC and accuracy (chance = 0.5) for: all bugs vs clean; op mutants vs clean;
llm mutants vs clean. Features: token counts over the ADDED lines of the diff only (what a mutation
changes), with and without the mutated line's immediate tokens. Also lists the top-weighted tokens,
which is the human-readable version of the leak (e.g. ' or ' / ' and ' dominating = operator shapes).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def added_text(diff: str) -> str:
    return "\n".join(l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_]\w*|==|!=|<=|>=|->|[^\sA-Za-z_0-9]", text)


def cv_auroc(X, y, folds=5, seed=0) -> tuple[float, float, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    pred = np.zeros(len(y)); coefs = np.zeros(X.shape[1])
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]; coefs += clf.coef_[0]
    return roc_auc_score(y, pred), float(((pred > 0.5) == y).mean()), coefs / folds


def run(tasks, pos_filter, label):
    from sklearn.feature_extraction.text import CountVectorizer
    rows = [t for t in tasks if t["kind"] == "clean" or pos_filter(t)]
    y = np.array([1 if t["kind"] == "bug" else 0 for t in rows])
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        print(f"{label:28s} too few rows ({y.sum()} bug / {len(y) - y.sum()} clean)"); return
    docs = [" ".join(tokens(added_text(t["diff"]))) for t in rows]
    vec = CountVectorizer(token_pattern=r"\S+", min_df=2, binary=True)
    X = vec.fit_transform(docs).toarray().astype(float)
    auc, acc, coef = cv_auroc(X, y)
    names = vec.get_feature_names_out()
    top = sorted(zip(coef, names), reverse=True)[:8]
    print(f"{label:28s} n={len(y):3d} ({y.sum()} bug) AUROC={auc:.2f} acc={acc:.2f}  top bug-tokens: " + ", ".join(f"{n}" for _, n in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="review/bench/public_tasks.jsonl")
    a = ap.parse_args()
    tasks = [json.loads(l) for l in Path(a.tasks).read_text().splitlines() if l.strip()]
    print("Diff-text-only linear probe, 5-fold CV. AUROC 0.5 = no surface leak; >0.75 = the bench is partly solvable without reading code.\n")
    run(tasks, lambda t: True, "all bugs vs clean")
    run(tasks, lambda t: t.get("source") == "op", "operator mutants vs clean")
    run(tasks, lambda t: t.get("source") == "llm", "llm mutants vs clean")
    run(tasks, lambda t: t.get("source") == "real", "real positives vs clean")
    print("\nNote: a mutant and its clean original share every added line but one, so a high AUROC here means the\n"
          "single mutated line carries a learnable signature (e.g. 'or' where 'and' is idiomatic). Compare op vs llm.")


if __name__ == "__main__":
    main()
