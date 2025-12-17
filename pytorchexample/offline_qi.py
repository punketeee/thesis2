# pytorchexample/offline_qi.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class OfflineQIConfig:
    events: Tuple[str, ...] = ("neg", "inc", "help")
    option: str = "count"          # match q_inf.py default use-case
    ignorefirst: int = 0
    ignorelast: int = 0
    threshold: float = 0.0         # treshold in q_inf.py (kept same spelling there)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_test_improvements(acc: List[float]) -> List[float]:
    """
    Convert accuracy-per-round into improvement-per-round.
    We return a list imp with same indexing as rounds:
      imp[0] = 0
      imp[r] = acc[r] - acc[r-1]
    """
    if len(acc) < 2:
        return [0.0]
    imp = [0.0]
    for r in range(1, len(acc)):
        imp.append(acc[r] - acc[r - 1])
    return imp


def offline_qi_score(
    contributors: Dict[int, List[int]],
    test_imp: List[float],
    cfg: OfflineQIConfig,
) -> Dict[int, Dict[int, int]]:
    """
    Re-implements q_inf.py 'test(...)' logic for option='count',
    BUT uses dicts instead of allocating score[round, max_client_id+1].

    Returns:
      scores_by_round[r][node_id] = cumulative score at round r (0-indexed in output rounds)
    """
    # Determine available rounds for scoring
    # In your logs contributors are keyed by 1..R, and test_imp is indexed 0..R
    R = len(test_imp) - 1
    start_round = cfg.ignorefirst
    end_round = R - cfg.ignorelast

    # We'll build a cumulative score dict per round (like q_inf.py does: it updates score[r..end])
    scores_by_round: Dict[int, Dict[int, int]] = {}
    current: Dict[int, int] = {}

    def apply_to_all_future(round_idx: int, node_ids: List[int], delta: int) -> None:
        nonlocal current
        # In q_inf.py, it adds starting from 'round' to the end.
        # We'll simulate that by updating current now, and snapshotting each round below.
        for nid in node_ids:
            current[nid] = current.get(nid, 0) + delta

    for r in range(0, end_round + 1):
        # snapshot BEFORE possible update at r? In q_inf.py, updates happen for round=r
        # and are written into score[r], score[r+1], ... . So snapshot after applying events at r.
        if r >= start_round:
            node_ids = contributors.get(r, [])  # r=0 usually empty; your dict starts at 1

            if "neg" in cfg.events:
                if test_imp[r] < -cfg.threshold:
                    apply_to_all_future(r, node_ids, -1)

            if "inc" in cfg.events:
                if r <= end_round - 1 and test_imp[r] < (test_imp[r + 1] - cfg.threshold):
                    apply_to_all_future(r, node_ids, -1)

            if "help" in cfg.events:
                if r >= 1 and test_imp[r] > (test_imp[r - 1] + cfg.threshold):
                    apply_to_all_future(r, node_ids, +1)

        scores_by_round[r] = dict(current)

    return scores_by_round


def run_offline_qi(
    qi_dir: str = "qi_logs",
    out_path: str = "qi_logs/offline_qi_scores.json",
    cfg: OfflineQIConfig = OfflineQIConfig(),
) -> None:
    qi_path = Path(qi_dir)

    node_id_map_path = qi_path / "node_id_map.json"
    node_id_map = _load_json(node_id_map_path) if node_id_map_path.exists() else {}

    contrib_raw = _load_json(qi_path / "contributors.json")
    metrics = _load_json(qi_path / "global_metrics.json")


    # Parse contributors: JSON keys are strings -> int
    contributors: Dict[int, List[int]] = {int(k): v for k, v in contrib_raw.items()}

    acc: List[float] = metrics["accuracy"]
    test_imp = compute_test_improvements(acc)

    scores_by_round = offline_qi_score(contributors, test_imp, cfg)

    out = {
        "config": {
            "events": list(cfg.events),
            "option": cfg.option,
            "ignorefirst": cfg.ignorefirst,
            "ignorelast": cfg.ignorelast,
            "threshold": cfg.threshold,
        },
        "accuracy": acc,
        "test_improvement": test_imp,
        "scores_by_round": {str(k): v for k, v in scores_by_round.items()},
        "final_scores": scores_by_round[max(scores_by_round.keys())] if scores_by_round else {},
    }

    Path(out_path).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[offline_qi] Wrote: {out_path}")


if __name__ == "__main__":
    run_offline_qi()
