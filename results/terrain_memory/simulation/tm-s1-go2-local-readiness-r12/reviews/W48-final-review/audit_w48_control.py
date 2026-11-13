#!/usr/bin/env python3
"""Audit committed W48 control evidence; stdlib only, no physics or network."""
import argparse
import csv
import hashlib
import io
import json
import pathlib
import re
from collections import Counter


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def apply_one(original, patch):
    old = original.splitlines(keepends=True)
    lines = patch.splitlines(keepends=True)
    out, pos, i = [], 0, 0
    while i < len(lines):
        if not lines[i].startswith("@@"):
            i += 1
            continue
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[i])
        assert m
        start = int(m[1]) - 1
        out.extend(old[pos:start])
        pos = start
        i += 1
        while i < len(lines) and not lines[i].startswith(("@@", "--- ")):
            line = lines[i]
            if line[:1] in (" ", "-"):
                assert old[pos] == line[1:], (pos, line)
                if line[0] == " ":
                    out.append(old[pos])
                pos += 1
            elif line[:1] == "+":
                out.append(line[1:])
            else:
                raise ValueError(line)
            i += 1
    out.extend(old[pos:])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=pathlib.Path, required=True)
    ap.add_argument("--r35-source", type=pathlib.Path)
    ap.add_argument("--rpc-source", type=pathlib.Path,
                    help="Optional reconstructed r36 grpc_agent_util.cc; compare its SHA.")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    a = ap.parse_args()
    root = a.repo
    base = root / "results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12"
    w48 = base / "w48"
    sources = []

    def read(p):
        b = p.read_bytes()
        sources.append(dict(path=str(p.relative_to(root)), bytes=len(b), sha256=sha(b)))
        return json.loads(b)

    build = read(w48 / "build.json")
    overall = read(w48 / "task-a/summary.json")
    algebra = read(w48 / "task-a/algebra.json")
    startup = read(w48 / "startup-technical.json")
    compare = read(w48 / "original-comparison.json")
    total_work = Counter()
    rows = []
    policy_pairs = positive_pairs_changed = 0
    cost_error = 0.0
    max_positive_change = 0.0
    all_candidate_lookup = {}
    for label in ["A148", "A153", "B155", "B158"]:
        old = read(w48 / "task-a" / (label + "-old.json"))
        new = read(w48 / "task-a" / (label + "-corrected.json"))
        dep = read(w48 / "task-a" / (label + "-deployment.json"))
        assert old["identity"] == new["identity"]
        assert old["selected_policy_sha256"] == new["selected_policy_sha256"]
        assert old["installed_policy_identity"] == new["installed_policy_identity"]
        for field in ["task_parameters", "cost_weights", "mode_readback"]:
            assert old[field] == new[field]
        for variant, d in [("old", old), ("corrected", new)]:
            assert d["status"] == "completed"
            assert d["actual_exe_sha256"] == d["binary_sha256"]
            assert d["call"]["r33_mechanism"] == "N"
            assert d["call"]["r34_enabled"]
            assert d["call"]["r35_branch"] == "constant"
            assert d["call"]["stop_reason"] == "native_search_stagnation"
            assert len(d["candidates"]) == 60
            total_work.update(d["work"])
            for c in d["candidates"]:
                cost_error = max(cost_error, abs(sum(c["J_parts"].values()) - c["native_J"]))
                assert not c["failure"] and c["trunk_violation"] == 0
                if c["kind"] == 1:
                    all_candidate_lookup[(label, variant, int(c["slot"] / 10) + 1, c["alpha"])] = c
        assert new["binary_sha256"] == build["binary_sha256"]
        assert old["binary_sha256"] == build["prior_binary_sha256"]
        for x, y in zip(old["call"]["search_attempts"], new["call"]["search_attempts"]):
            for key in ["dV_unscaled", "Qu_unregularized_max", "projected_gradient_inf",
                        "k_unscaled_max", "K_max", "lambda_used", "rate_used"]:
                assert x[key] == y[key], (label, key)
        categories = {}
        zero_delta = []
        for name, predicate in [
            ("nominal", lambda c: c["kind"] == 0),
            ("positive", lambda c: c["kind"] == 1 and c["alpha"] > 0),
            ("zero", lambda c: c["kind"] == 1 and c["alpha"] == 0),
        ]:
            pairs = [(x, y) for x, y in zip(old["candidates"], new["candidates"]) if predicate(x)]
            assert all(x["policy_identity"] == y["policy_identity"] for x, y in pairs)
            policy_pairs += len(pairs)
            changes = [y["native_J"] - x["native_J"] for x, y in pairs]
            categories[name] = dict(count=len(pairs), changed_cost_count=sum(z != 0 for z in changes),
                                    max_abs_cost_change=max(map(abs, changes)),
                                    changed_serialized_policy_count=0)
            if name == "positive":
                mins = {}
                for v, d in [("old", old), ("corrected", new)]:
                    pp = [c for c in d["candidates"] if predicate(c)]
                    delta = [c["native_J"] - d["call"]["search_attempts"][int(c["slot"] / 10)]["nominal_current_cost"] for c in pp]
                    assert min(delta) > 0
                    mins[v] = min(delta)
                categories[name]["minimum_actual_delta_J"] = mins
                positive_pairs_changed += categories[name]["changed_cost_count"]
                max_positive_change = max(max_positive_change, categories[name]["max_abs_cost_change"])
            if name == "zero":
                for i, (x, y) in enumerate(pairs):
                    dx = x["native_J"] - old["call"]["search_attempts"][i]["nominal_current_cost"]
                    dy = y["native_J"] - new["call"]["search_attempts"][i]["nominal_current_cost"]
                    assert dy == 0
                    zero_delta.append(dict(attempt=i + 1, old_delta_J=dx, corrected_delta_J=dy))
        assert dep["deployment_comparison"]["score_path"] == "ActionImpl/Rollout at actual recorded scale"
        assert all(v == 0 for v in dep["deployment_comparison"]["max_abs_errors"].values())
        assert dep["deployment_comparison"]["delta_J"] == 0
        assert dep["work"]["physics_substeps"] == 1750
        assert dep["work"]["action_queries_no_integration"] == 175
        for key in overall["work"]:
            total_work[key] += dep["work"].get(key, 0)
        rows.append(dict(label=label, selected_source=new["call"]["candidate_source"],
                         selected_policy_sha256=new["selected_policy_sha256"],
                         selected_identity=new["installed_policy_identity"],
                         scale=new["actual_scale"], selected_J=new["selected"]["native_J"],
                         dx_m=new["selected"]["dx_m"], zero_delta_J=zero_delta,
                         categories=categories, direction_summaries_exact_equal=True,
                         actual_scale_deployment_max_errors=dep["deployment_comparison"]["max_abs_errors"]))
    assert policy_pairs == 240
    assert positive_pairs_changed == 98
    for key, val in overall["work"].items():
        assert total_work[key] == val, (key, total_work[key], val)
    assert overall["native_rollout_physics_substeps"] == 484 * 175 * 10
    cp = w48 / "task-a/positive-directions.csv"
    cb = cp.read_bytes()
    sources.append(dict(path=str(cp.relative_to(root)), bytes=len(cb), sha256=sha(cb)))
    csv_rows = list(csv.DictReader(io.StringIO(cb.decode())))
    assert len(csv_rows) == 400
    formula_error = 0.0
    positive = Counter()
    for c in csv_rows:
        alpha = float(c["alpha"])
        delta = float(c["candidate_J"]) - float(c["nominal_J"])
        pred = -alpha * float(c["dV0"]) - alpha * alpha * float(c["dV1"])
        formula_error = max(formula_error, abs(delta-float(c["actual_delta_J"])),
                            abs(pred-float(c["predicted_decrease"])))
        key = (c["label"], c["variant"], int(c["attempt"]), alpha)
        assert all_candidate_lookup[key]["native_J"] == float(c["candidate_J"])
        if alpha > 0:
            assert pred > 0 and delta > 0
            positive[c["variant"]] += 1
    assert positive == dict(old=180, corrected=180)
    assert formula_error == 0 and cost_error < 1e-12
    for fixture in algebra["rows"]:
        values = {r["case"]: r["ds"] for r in fixture["results"]}
        assert not any(values[0]) and not any(values[1])
        assert abs(values[2][3] - 1e-8) < 1e-16
        assert abs(values[3][3] + 1e-8) < 1e-16
        assert abs(values[4][0] - .001) < 1e-16
        assert abs(values[4][18] - .002) < 1e-16
    assert algebra["integration_steps"] == 0 and algebra["optimizer_solves"] == 0
    activation = {r["case"]: r["ds"] for r in algebra["activation_model"]}
    assert abs(activation[4][-1] - .003) < 1e-16
    patch_path = root / "experiments/terrain_memory/s1-go2-r11-S5-r36.patch"
    patch = patch_path.read_text()
    assert sha(patch.encode()) == build["patch_sha256"]
    patch_checks = []
    chunks = re.split(r"(?=^--- a/)", patch, flags=re.M)
    for chunk in chunks:
        if not chunk.strip():
            continue
        rel = chunk.splitlines()[0][6:]
        result = dict(path=rel, expected_sha256=build["source_files_sha256"][rel])
        if a.r35_source:
            applied = apply_one((a.r35_source / rel).read_text(), chunk)
            result.update(reconstructed_sha256=sha(applied.encode()), reconstructed=True)
            assert result["reconstructed_sha256"] == result["expected_sha256"]
        patch_checks.append(result)
    assert len(patch_checks) == 2
    assert startup["original_plant_physics_substeps"] == 0
    assert len(startup["technical_attempts"]) == 4
    assert "r12_trunk_clearance" in startup["restored_configuration_keys"]
    assert "r12_trunk_clearance" in build["configuration_overrides"]
    formal = []
    for case, seed in [("a", 1), ("b", 2)]:
        resolved = read(base / f"job-20261113T090632Z-w48-{case}-eval-PANN_PPO_S5-s{seed}-E30-attempt01-resolved.json")
        native = resolved["w48_native_process"]
        assert native["exe_sha256"] == build["binary_sha256"]
        env = native["environment"]
        assert env["MJPC_R33_MECHANISM"] == "N"
        assert env["MJPC_R34_ACCEPTANCE"] == "strict"
        assert env["MJPC_R35_BRANCH"] == "constant"
        for forbidden in ["MJPC_R34_SCORE_ONLY", "MJPC_W47_INDEXED_SCORE", "MJPC_R33_INPUT_POLICY", "MJPC_R34_CONTEXT", "MJPC_R35_REFERENCE"]:
            assert forbidden not in env
        formal.append(dict(case=case.upper(), binary_sha256=native["exe_sha256"],
                           model_mjb_sha256=resolved["model_mjb_sha256"],
                           mechanism="N", acceptance="strict", branch="constant"))
    rpc = dict(scope="Source path audit, not a new RPC measurement.",
               explicit_time_in_runtime=True,
               inherited_proto_time="optional float", inherited_proto_action="repeated float",
               proto_sha256=sha((root/"experiments/terrain_memory/s1-go2-r25-source/mjpc/grpc/agent.proto").read_bytes()),
               proto_not_listed_separately_in_r36_source_manifest=True,
               implicit_time_uses_native_state=True,
               response_rounding_known_in_W40=True,
               actual_effect_on_failures_not_established=True)
    if a.rpc_source:
        rb = a.rpc_source.read_bytes()
        expected = build["source_files_sha256"]["mjpc/grpc/grpc_agent_util.cc"]
        assert sha(rb) == expected
        assert "request->has_time() ? request->time() : agent->state.time()" in rb.decode()
        rpc.update(reconstructed_source_sha256=sha(rb), matches_r36_build_manifest=True)
    out = dict(schema="w48_committed_control_review_v1",
               reviewed_commit="61e16975fa6b38dc8e2d31779c2bd82f1aaeb8b6",
               new_optimizer_or_physics_work=0,
               counts=dict(solves=8, positive_candidates_old=180, positive_candidates_corrected=180,
                           positive_pairs_cost_changed=positive_pairs_changed,
                           matched_serialized_policy_pairs=policy_pairs,
                           zero_pairs_repaired=20, deployment_rollouts=4),
               max_positive_cost_change=max_positive_change, max_cost_parts_error=cost_error,
               max_csv_formula_error=formula_error, finite_work=dict(total_work), rows=rows,
               patch_checks=patch_checks, formal_processes=formal, startup_technical=startup,
               original_trajectory_comparison_from_committed_summary=compare,
               rpc=rpc, sources=sources,
               limitations=[
                   "Original trajectory/policy binary payloads were not reread; same-policy and original trajectory equality use committed hashes and executor comparison.",
                   "Policy identity hashes arrays/reference/gains/metric/scales, not executable mapping semantics, which differ by StateDiff fix.",
                   "Four deployment checks all selected nominal/incumbent ActionImpl paths; they do not test a positive winner across ActionIndex→ActionImpl or the end-to-end RPC loop.",
                   "Finite action queries are observation-only after the fixed rollout; float RPC results are not integrated in those checks.",
                   "Whole-task chosen_actual differs in 123 A and 38 B calls despite identical selected-source/alpha and plant arrays; this is not proof every discarded candidate is unchanged.",
               ])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:out[k] for k in ["counts","max_positive_cost_change","finite_work","patch_checks","rpc"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
