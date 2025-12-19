from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float_map(d: Dict) -> Dict[str, float]:
    """Convert a json dict of scores into {client_id(str): float} robustly."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        # keep client ids as strings for consistent keys across json/csv
        ks = str(k)
        try:
            out[ks] = float(v)
        except Exception:
            # if something weird appears, treat as NaN
            out[ks] = float("nan")
    return out

def extract_qi_final_scores(score_obj: Any, node_id_map_path: Path) -> Dict[str, float]:
    """
    Returns {client_id(str): float} for ONE seed by:
      - reading score_obj["final_scores"] (keyed by node_id)
      - mapping node_id -> client_id using node_id_map.json

    Example:
      final_scores: {"6540...": -9, ...}
      node_id_map : {"6540...": 3, ...}
      -> {"3": -9, ...}
    """
    if not isinstance(score_obj, dict):
        raise ValueError("offline_qi_scores.json must be a dict")

    if "final_scores" not in score_obj or not isinstance(score_obj["final_scores"], dict):
        raise KeyError("offline_qi_scores.json missing 'final_scores' dict")

    node_to_client = load_json(node_id_map_path)
    if not isinstance(node_to_client, dict):
        raise ValueError("node_id_map.json must be a dict")

    # normalize map keys/values to strings
    node_to_client_str = {str(node): str(cid) for node, cid in node_to_client.items()}

    out: Dict[str, float] = {}
    for node_id, val in score_obj["final_scores"].items():
        node_id_str = str(node_id)
        if node_id_str not in node_to_client_str:
            # if this happens, something is inconsistent in logging
            continue
        client_id_str = node_to_client_str[node_id_str]
        try:
            out[client_id_str] = float(val)
        except Exception:
            out[client_id_str] = float("nan")

    return out


def extract_qi_scores(score_obj: Any, node_id_map_path: Path) -> Dict[str, float]:
    """
    Convert one seed's offline_qi_scores.json into {client_id(str): float}.

    In your logs:
      - score_obj["final_scores"] is keyed by Flower node_id (big ints as strings)
      - node_id_map.json maps node_id -> logical client_id (0..4)

    We map node_id -> client_id so that seeds align and you get n=10 per client.
    """
    if not isinstance(score_obj, dict):
        raise ValueError("offline_qi_scores.json must be a dict")

    if "final_scores" not in score_obj or not isinstance(score_obj["final_scores"], dict):
        raise KeyError("offline_qi_scores.json missing 'final_scores' dict")

    node_to_client = load_json(node_id_map_path)
    if not isinstance(node_to_client, dict):
        raise ValueError("node_id_map.json must be a dict")

    # normalize mapping: node_id(str) -> client_id(str)
    node_to_client = {str(node): str(cid) for node, cid in node_to_client.items()}

    out: Dict[str, float] = {}
    for node_id, val in score_obj["final_scores"].items():
        node_id = str(node_id)
        if node_id not in node_to_client:
            continue
        client_id = node_to_client[node_id]
        try:
            out[client_id] = float(val)
        except Exception:
            out[client_id] = float("nan")

    return out



def aggregate_experiment(exp_dir: Path, seeds: List[int]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    """
    Returns:
      mean_by_client, std_by_client, n_by_client (#seeds used per client)
    """
    per_seed_scores: List[Dict[str, float]] = []

    found_any = False
    for s in seeds:
        score_path = exp_dir / f"seed_{s:02d}" / "qi" / "offline_qi_scores.json"
        if not score_path.exists():
            continue
        found_any = True
        
        obj = load_json(score_path)
        node_map_path = exp_dir / f"seed_{s:02d}" / "qi" / "node_id_map.json"
        scores = extract_qi_scores(obj, node_map_path)
        per_seed_scores.append(scores)


    if not found_any:
        return {}, {}, {}

    # union of all clients appearing in any seed
    clients = sorted({cid for sd in per_seed_scores for cid in sd.keys()})

    mean_by_client: Dict[str, float] = {}
    std_by_client: Dict[str, float] = {}
    n_by_client: Dict[str, int] = {}

    for cid in clients:
        vals = []
        for sd in per_seed_scores:
            if cid in sd:
                v = sd[cid]
                if not (isinstance(v, float) and math.isnan(v)):
                    vals.append(v)
        n = len(vals)
        n_by_client[cid] = n
        if n == 0:
            mean_by_client[cid] = float("nan")
            std_by_client[cid] = float("nan")
        elif n == 1:
            mean_by_client[cid] = float(vals[0])
            std_by_client[cid] = 0.0
        else:
            arr = np.array(vals, dtype=np.float64)
            mean_by_client[cid] = float(arr.mean())
            # sample std (ddof=1) is typical for reporting across seeds
            std_by_client[cid] = float(arr.std(ddof=1))

    return mean_by_client, std_by_client, n_by_client


def write_csv(path: Path, mapping: Dict[str, float]) -> None:
    lines = ["client_id,value"]
    for cid, val in mapping.items():
        lines.append(f"{cid},{val}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"

    if not results_dir.exists():
        raise FileNotFoundError(f"Cannot find results/ at: {results_dir}")

    seeds = list(range(10))  # seed_00 ... seed_09

    exp_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()])
    if not exp_dirs:
        print(f"No experiment folders found under {results_dir}")
        return

    print(f"Found {len(exp_dirs)} experiment folders under: {results_dir}")

    for exp_dir in exp_dirs:
        mean_map, std_map, n_map = aggregate_experiment(exp_dir, seeds)
        if not mean_map:
            print(f"  - {exp_dir.name}: SKIP (no offline_qi_scores.json found in any seed)")
            continue

        out_dir = exp_dir / "seed_agg"
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON summary (mean/std/n per client)
        summary = {
            "experiment": exp_dir.name,
            "seeds_expected": [f"seed_{s:02d}" for s in seeds],
            "client_stats": {
                cid: {"mean": mean_map[cid], "std": std_map[cid], "n": n_map.get(cid, 0)}
                for cid in sorted(mean_map.keys())
            },
        }
        (out_dir / "qi_scores_mean_std.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # CSVs for easy plotting
        write_csv(out_dir / "qi_scores_mean.csv", mean_map)
        write_csv(out_dir / "qi_scores_std.csv", std_map)

        # Quick console summary
        used_seeds = []
        for s in seeds:
            if (exp_dir / f"seed_{s:02d}" / "qi" / "offline_qi_scores.json").exists():
                used_seeds.append(s)
        print(f"  - {exp_dir.name}: aggregated (seeds found: {len(used_seeds)}/10) -> {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
