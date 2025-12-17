# pytorchexample/qi_fedavg.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import json
import torch
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


def _f(x: object, default: float = 0.0) -> float:
    try:
        return float(x)  # MetricRecord values are JSON-ish
    except Exception:
        return default


@dataclass
class QILog:
    out_dir: Path = Path("qi_logs")
    contributors: Dict[int, List[int]] = field(default_factory=dict)  # round -> [node_id]
    acc: List[float] = field(default_factory=list)                   # index = server_round
    loss: List[float] = field(default_factory=list)

    def dump(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "contributors.json").write_text(
            json.dumps(self.contributors, indent=2, sort_keys=True)
        )
        (self.out_dir / "global_metrics.json").write_text(
            json.dumps({"accuracy": self.acc, "loss": self.loss}, indent=2)
        )


class QIFedAvg(FedAvg):
    """
    FedAvg + original-project-style Quality Inference (QI) *online* weighting.

    What it does (matching the old repo):
      - maintain per-client quality weights q_i (start at 1.0)
      - after each centralized evaluation, compute test_improvement
      - update q_i using the same reward/penalty logic
      - in aggregate_train, before aggregating, replace each client model w_k with:
            w'_k = w_global + (w_k - w_global) * mean(q_i for i in contributors_of_round)
        (note: mean over participants, same scalar applied to all participants)
    """

    def __init__(
        self,
        *,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        qi_step: float = 0.05,              # like args.weight in old repo
        qi_min: float = 0.10,               # clamp to avoid killing updates
        qi_max: float = 5.0,                # clamp to avoid exploding updates
        qi_out_dir: str = "qi_logs",
        **kwargs,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            **kwargs,
        )
        self.qi_step = float(qi_step)
        self.qi_min = float(qi_min)
        self.qi_max = float(qi_max)

        self.log = QILog(out_dir=Path(qi_out_dir))

        # q_i weights (node_id -> weight)
        self.q: Dict[int, float] = {}

        # cached state needed for update logic
        self._current_global: Optional[ArrayRecord] = None
        self._last_acc: Optional[float] = None
        self._last_imp: Optional[float] = None
        self._last_contributors: Optional[List[int]] = None

    # ---- Helper: update q_i with the original rule --------------------------
    def _update_q(self, server_round: int, acc_now: float) -> None:
        # server_round is 0 for initial eval, then 1..R after each round
        if self._last_acc is None:
            self._last_acc = acc_now
            self._last_imp = None
            return

        imp = acc_now - self._last_acc  # test_improvement of this evaluation point
        self._last_acc = acc_now

        contrib = self._last_contributors or []

        # Rule 1: if test_improvement < 0 -> penalize contributors of this round
        if imp < 0.0:
            for nid in contrib:
                self.q[nid] = max(self.qi_min, self.q.get(nid, 1.0) * (1.0 - self.qi_step))

        # Rule 2: if this improvement > previous improvement:
        # reward current contributors; penalize previous-round contributors
        if self._last_imp is not None and imp > self._last_imp:
            for nid in contrib:
                self.q[nid] = min(self.qi_max, self.q.get(nid, 1.0) * (1.0 + self.qi_step))
            for nid in (self._last_contributors or []):
                self.q[nid] = max(self.qi_min, self.q.get(nid, 1.0) * (1.0 - self.qi_step))

        self._last_imp = imp

    # ---- Wrap evaluate_fn so we can update q_i online ------------------------
    def start(
        self,
        *,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None,
    ):
        # keep global arrays for scaling client updates
        self._current_global = initial_arrays

        if evaluate_fn is None:
            return super().start(
                grid=grid,
                initial_arrays=initial_arrays,
                num_rounds=num_rounds,
                timeout=timeout,
                train_config=train_config,
                evaluate_config=evaluate_config,
                evaluate_fn=None,
            )

        def wrapped_evaluate_fn(server_round: int, arrays: ArrayRecord) -> Optional[MetricRecord]:
            m = evaluate_fn(server_round, arrays)
            if m is None:
                return None

            acc = _f(m.get("accuracy", 0.0))
            loss = _f(m.get("loss", 0.0))

            # log it
            while len(self.log.acc) <= server_round:
                self.log.acc.append(0.0)
                self.log.loss.append(0.0)
            self.log.acc[server_round] = acc
            self.log.loss[server_round] = loss
            self.log.dump()

            # update q_i online using the original logic
            self._update_q(server_round, acc)
            return m

        return super().start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=num_rounds,
            timeout=timeout,
            train_config=train_config,
            evaluate_config=evaluate_config,
            evaluate_fn=wrapped_evaluate_fn,
        )

    # ---- Core: scale client updates before FedAvg aggregation ----------------
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        # Materialize replies because we’ll iterate multiple times
        replies_list = list(replies)

        # Contributors = src_node_id of each reply :contentReference[oaicite:2]{index=2}
        contributors = [msg.metadata.src_node_id for msg in replies_list]
        self.log.contributors[server_round] = contributors
        self.log.dump()

        # init q_i for new nodes
        for nid in contributors:
            self.q.setdefault(nid, 1.0)

        # store "last contributors" for update rule #2
        self._last_contributors = contributors

        # if we don’t have global arrays, just fall back
        if self._current_global is None:
            return super().aggregate_train(server_round, replies_list)

        # tmp_w = mean(q_i over this round’s contributors)  (same scalar for everyone)
        tmp_w = sum(self.q[nid] for nid in contributors) / max(len(contributors), 1)

        # scale each client ArrayRecord before calling FedAvg’s aggregation
        global_sd = self._current_global.to_torch_state_dict()
        new_replies: List[Message] = []

        for msg in replies_list:
            content = msg.content

            # FedAvg uses arrayrecord_key="arrays" by default :contentReference[oaicite:3]{index=3}
            arr: ArrayRecord = content[self.arrayrecord_key]  # type: ignore[attr-defined]
            client_sd = arr.to_torch_state_dict()

            scaled_sd = {}
            for k, g in global_sd.items():
                c = client_sd[k]
                # w' = g + (c - g) * tmp_w
                scaled_sd[k] = g + (c - g) * tmp_w

            content[self.arrayrecord_key] = ArrayRecord(scaled_sd)  # replace arrays
            new_replies.append(msg)

        arrays_agg, metrics_agg = super().aggregate_train(server_round, new_replies)

        # update cached global arrays for next round
        if arrays_agg is not None:
            self._current_global = arrays_agg

        return arrays_agg, metrics_agg
