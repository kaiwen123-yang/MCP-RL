#!/usr/bin/env python3
"""Derive S0 decision comparisons from already executed reference candidates.

The task-conditioned policy chooses one action retrospectively for each future
load/return-time family, averaging its six different prefixes. It does not use
the current prefix to choose an action. This is an optimistic descriptive
comparison on the same deterministic grid, not training or held-out evaluation.

Resolution sensitivity repeats full-information selection at the original
5e-8 m tolerance and hypothetical 1e-6/1e-5 m tolerances. The latter are neither
calibrated sensor resolutions nor changes to the original experiment.

Only candidates.csv, contexts.csv and actions.csv are read. No material
integration, model training, archive download or original-result mutation occurs.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PEAK = "max_remaining_pad_compression_m"
TIME = "completion_time_s"
WORK = "positive_remaining_contact_work_J"
ORIGINAL_TOLERANCE_M = 5e-8
BUDGETS_S = (0.0, 0.5, 1.5)
TOLERANCES_M = (ORIGINAL_TOLERANCE_M, 1e-6, 1e-5)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def average(values):
    return float(np.mean(values))


def choose(rows, tolerance):
    if not rows:
        return None
    minimum = min(row[PEAK] for row in rows)
    tied = [row for row in rows if row[PEAK] <= minimum + tolerance]
    return min(tied, key=lambda row: (row[TIME], row[WORK], row["candidate_id"]))


def aggregate(selected, oracles):
    complete = [row for row in selected if row is not None]
    pairs = [(row, oracle) for row, oracle in zip(selected, oracles)
             if row is not None and oracle is not None]
    result = {
        "contexts": len(selected),
        "selection_failures": len(selected) - len(complete),
        "paired_contexts": len(pairs),
        "statistics_scope": "selected feasible contexts; failure counts retained",
        "mean_peak_mm": average([row[PEAK] for row in complete]) * 1000,
        "max_peak_mm": max(row[PEAK] for row in complete) * 1000,
        "mean_time_s": average([row[TIME] for row in complete]),
        "mean_positive_contact_work_J": average([row[WORK] for row in complete]),
        "mean_extra_wait_s": average([row["wait_s"] for row in complete]),
        "zero_extra_wait_contexts": sum(row["wait_s"] == 0 for row in complete),
        "selected_action_counts": json.dumps(dict(sorted(Counter(
            row["candidate_id"] for row in complete).items())), sort_keys=True),
    }
    differences = [row[PEAK] - oracle[PEAK] for row, oracle in pairs]
    result.update({
        "mean_peak_minus_original_oracle_mm": average(differences) * 1000,
        "max_peak_minus_original_oracle_mm": max(differences) * 1000,
        "mean_time_minus_original_oracle_s": average([
            row[TIME] - oracle[TIME] for row, oracle in pairs]),
        "mean_work_minus_original_oracle_J": average([
            row[WORK] - oracle[WORK] for row, oracle in pairs]),
        "different_action_from_original_oracle": sum(
            row["candidate_id"] != oracle["candidate_id"] for row, oracle in pairs),
        "peak_worse_than_original_oracle_beyond_original_tolerance": sum(
            value > ORIGINAL_TOLERANCE_M for value in differences),
    })
    return result


def analyze(directory):
    source_digest = hashlib.sha256((directory / "candidates.csv").read_bytes()).hexdigest()
    contexts = read_csv(directory / "contexts.csv")
    for row in contexts:
        row["context_id"] = int(row["context_id"])
        for key in ("gamma", "prefix_gap_s", "hind_N", "return_N", "return_gap_s"):
            row[key] = float(row[key])
    contexts.sort(key=lambda row: row["context_id"])
    actions = read_csv(directory / "actions.csv")
    for row in actions:
        row["candidate_id"] = int(row["candidate_id"])
        row["wait_s"] = float(row["wait_s"])
    actions.sort(key=lambda row: row["candidate_id"])

    by_context = defaultdict(list)
    lookup = {}
    for row in read_csv(directory / "candidates.csv"):
        if row["predictor"] != "reference_continuous":
            continue
        row["context_id"] = int(row["context_id"])
        row["candidate_id"] = int(row["candidate_id"])
        for key in (PEAK, TIME, WORK, "wait_s"):
            row[key] = float(row[key])
        row["remaining_feasible"] = row["remaining_feasible"] == "True"
        by_context[row["context_id"]].append(row)
        lookup[row["context_id"], row["candidate_id"]] = row

    policy_rows, summaries, resolutions = [], [], []
    for gamma in sorted({row["gamma"] for row in contexts}, reverse=True):
        gamma_contexts = [row for row in contexts if row["gamma"] == gamma]
        families = defaultdict(list)
        for context in gamma_contexts:
            families[context["hind_N"], context["return_N"], context["return_gap_s"]].append(context)
        for budget in BUDGETS_S:
            eligible = {context["context_id"]: [row for row in by_context[context["context_id"]]
                        if row["remaining_feasible"] and row["wait_s"] <= budget]
                        for context in gamma_contexts}
            oracle = {cid: choose(rows, ORIGINAL_TOLERANCE_M) for cid, rows in eligible.items()}
            family_choices = {}
            for key, family in sorted(families.items()):
                options = []
                for action in actions:
                    if action["wait_s"] > budget:
                        continue
                    rows = [lookup[context["context_id"], action["candidate_id"]] for context in family]
                    if all(row["remaining_feasible"] for row in rows):
                        options.append({"candidate_id": action["candidate_id"],
                                        PEAK: average([row[PEAK] for row in rows]),
                                        TIME: average([row[TIME] for row in rows]),
                                        WORK: average([row[WORK] for row in rows])})
                family_choices[key] = choose(options, ORIGINAL_TOLERANCE_M)

            selected, oracle_rows = [], []
            for context in gamma_contexts:
                cid = context["context_id"]
                family_key = (context["hind_N"], context["return_N"], context["return_gap_s"])
                family_choice = family_choices[family_key]
                actual = (None if family_choice is None else
                          lookup[cid, family_choice["candidate_id"]])
                original = oracle[cid]
                selected.append(actual)
                oracle_rows.append(original)
                policy_rows.append({
                    **context, "wait_budget_s": budget,
                    "family_prefix_count": len(families[family_key]),
                    "selection_scope": "retrospective future-conditioned action; prefix hidden; no holdout",
                    "selection_tolerance_m": ORIGINAL_TOLERANCE_M,
                    "selected_id": None if actual is None else actual["candidate_id"],
                    "decision_failure": actual is None,
                    "actual_peak_mm": None if actual is None else actual[PEAK] * 1000,
                    "actual_time_s": None if actual is None else actual[TIME],
                    "actual_positive_contact_work_J": None if actual is None else actual[WORK],
                    "actual_extra_wait_s": None if actual is None else actual["wait_s"],
                    "family_mean_peak_mm": None if family_choice is None else family_choice[PEAK] * 1000,
                    "oracle_selected_id": None if original is None else original["candidate_id"],
                    "oracle_peak_mm": None if original is None else original[PEAK] * 1000,
                    "oracle_time_s": None if original is None else original[TIME],
                    "peak_minus_oracle_mm": None if actual is None or original is None else (actual[PEAK] - original[PEAK]) * 1000,
                    "time_minus_oracle_s": None if actual is None or original is None else actual[TIME] - original[TIME],
                })
            summaries.append({
                "gamma": gamma, "wait_budget_s": budget,
                "selection_tolerance_m": ORIGINAL_TOLERANCE_M,
                "future_families": len(families),
                "comparison_scope": "retrospective descriptive comparison; no held-out generalization",
                "source_candidates_sha256": source_digest,
                **aggregate(selected, oracle_rows),
            })
            for tolerance in TOLERANCES_M:
                resolution_choices = [choose(eligible[context["context_id"]], tolerance)
                                      for context in gamma_contexts]
                resolutions.append({
                    "gamma": gamma, "wait_budget_s": budget,
                    "selection_tolerance_m": tolerance,
                    "selection_tolerance_um": tolerance * 1e6,
                    "tolerance_status": ("original protocol" if tolerance == ORIGINAL_TOLERANCE_M
                                         else "hypothetical; not calibrated sensor or task requirement"),
                    "comparison_scope": "full-information reselection of existing candidates; no new physics",
                    "source_candidates_sha256": source_digest,
                    **aggregate(resolution_choices, oracle_rows),
                })
    outputs = {
        "task-conditioned-policy.csv": policy_rows,
        "task-conditioned-summary.csv": summaries,
        "resolution-sensitivity.csv": resolutions,
    }
    for filename, rows in outputs.items():
        write_csv(directory / filename, rows)
    print(json.dumps({"result_directory": str(directory),
                      "new_physics_predictions": 0, "new_training_runs": 0,
                      "source_candidates_sha256": source_digest,
                      "output_rows": {filename: len(rows) for filename, rows in outputs.items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path(__file__).resolve().parents[1] /
                        "results/terrain_memory/simulation/tm-s0-contact-oracle-20261023")
    analyze(parser.parse_args().result_dir)
