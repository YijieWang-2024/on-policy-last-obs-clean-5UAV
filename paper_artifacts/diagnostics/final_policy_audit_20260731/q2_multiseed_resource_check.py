import json

import analyze_raw_actions as audit


SEEDS = [711954, 755766, 189921, 333333, 409098,
         169434, 740921, 515804, 916824, 938276]


rows = []
for seed in SEEDS:
    audit.SEED = seed
    learned = audit.evaluate("q2", audit.RUNS["q2"], "learned")
    equal_selected = audit.evaluate("q2", audit.RUNS["q2"], "equal_selected")
    rows.append({
        "seed": seed,
        "learned_true60": learned["true60"],
        "equal_selected_true60": equal_selected["true60"],
        "delta": equal_selected["true60"] - learned["true60"],
        "learned_completion_ratio": learned["completion_ratio"],
        "equal_selected_completion_ratio": equal_selected["completion_ratio"],
        "candidate_offloads": learned["candidate_offloads"],
        "selected_offloads": learned["selected_offloads"],
        "postprocess_cancel_ratio": learned["postprocess_cancel_ratio"],
    })

summary = {
    "seeds": SEEDS,
    "learned_true60_mean": sum(r["learned_true60"] for r in rows) / len(rows),
    "equal_selected_true60_mean": sum(r["equal_selected_true60"] for r in rows) / len(rows),
    "delta_mean": sum(r["delta"] for r in rows) / len(rows),
    "equal_selected_wins": sum(r["delta"] > 0 for r in rows),
    "rows": rows,
}
(audit.ROOT / "q2_multiseed_resource_check.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
