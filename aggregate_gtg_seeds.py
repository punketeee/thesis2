from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float_map(d: Dict) -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            out[str(k)] = float("nan")
    return out


def collapse_rounds(seed_gtg_dir: Path, field: str = "shapley_all") -> Dict[str, float]:
    """Sum shapley values over all rounds for one seed."""
    round_files = sorted(seed_gtg_dir.glob("round_*.json"))
    if not round_files:
        return {}

    per_round = []
    for rf in round_files:
        obj = load_json(rf)
        if field not in obj:
            raise KeyError(f"{rf} missing '{field}'")
        per_round.append(to_float_map(obj[field]))

    clients = sorted({cid for r in per_round for cid in r.keys()})
    out = {}

    for cid in clients:
        vals = []
        for r in per_round:
            if cid in r and not math.isnan(r[cid]):
                vals.append(r[cid])
        out[cid] = float(np.sum(vals)) if vals else float("nan")

    return out


def aggregate_seeds(root: Path, seeds: List[int]) -> Dict[str, Dict[str, float]]:
    per_seed_scores = []

    for s in seeds:
        seed_dir = root / f"seed_{s:02d}" / "gtg"
        if not seed_dir.exists():
            continue

        seed_score = collapse_rounds(seed_dir, field="shapley_all")
        if seed_score:
            per_seed_scores.append(seed_score)

    clients = sorted({cid for sd in per_seed_scores for cid in sd.keys()})
    mean, std, n = {}, {}, {}

    for cid in clients:
        vals = [
            sd[cid] for sd in per_seed_scores
            if cid in sd and not math.isnan(sd[cid])
        ]
        n[cid] = len(vals)

        if len(vals) == 0:
            mean[cid] = float("nan")
            std[cid] = float("nan")
        elif len(vals) == 1:
            mean[cid] = vals[0]
            std[cid] = 0.0
        else:
            arr = np.array(vals, dtype=np.float64)
            mean[cid] = float(arr.mean())
            std[cid] = float(arr.std(ddof=1))

    return {
        "mean": mean,
        "std": std,
        "n": n
    }


def main():
    project_root = Path(__file__).resolve().parent
    gtg_root = project_root / "qi_vs_gtg"

    seeds = list(range(10))

    agg = aggregate_seeds(gtg_root, seeds)

    out_dir = gtg_root / "seed_agg"
    out_dir.mkdir(exist_ok=True)

    out_json = {
        "experiment": "qi_vs_gtg",
        "round_reduction": "sum",
        "field": "shapley_all",
        "clients": {
            cid: {
                "mean": agg["mean"][cid],
                "std": agg["std"][cid],
                "n": agg["n"][cid]
            }
            for cid in sorted(agg["mean"].keys())
        }
    }

    with (out_dir / "gtg_scores_mean_std.json").open("w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print(f"GTG aggregation done. Results written to {out_dir}")


if __name__ == "__main__":
    main()
