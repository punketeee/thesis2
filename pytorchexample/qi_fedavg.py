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
    node_id_map: Dict[str, int] = field(default_factory=dict)  # str(node_id) -> partition-id

    def dump(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "contributors.json").write_text(
            json.dumps(self.contributors, indent=2, sort_keys=True)
        )
        (self.out_dir / "global_metrics.json").write_text(
            json.dumps({"accuracy": self.acc, "loss": self.loss}, indent=2)
        )
        (self.out_dir / "node_id_map.json").write_text(
            json.dumps(self.node_id_map, indent=2, sort_keys=True)
        )


class QIFedAvg(FedAvg):
    """
    FedAvg + logging for offline Quality Inference (QI).

    This strategy does NOT apply QI-weighted aggregation and does NOT update q_i online.
    It only logs:
      - contributors per round (Flower node_id)
      - centralized global metrics per round (accuracy/loss)
      - node_id -> partition-id mapping (so outputs are interpretable)
    Offline QI scoring is then computed from qi_logs/ by offline_qi.py.
    """

    def __init__(
        self,
        *,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        qi_out_dir: str = "qi_logs",
        **kwargs,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            **kwargs,
        )
        self.log = QILog(out_dir=Path(qi_out_dir))

        # q_i weights (node_id -> weight)
        self.q: Dict[int, float] = {}

        # cached state needed for update logic
        self._current_global: Optional[ArrayRecord] = None
        self._last_acc: Optional[float] = None
        self._last_imp: Optional[float] = None
       # self._last_contributors: Optional[List[int]] = None
        # contributors per round (server_round -> [node_id])
        self._contributors_by_round: Dict[int, List[int]] = {}



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
            #self._update_q(server_round, acc)
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

# ---- Core: FedAvg aggregation (QI logging only) ------------------------
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate client updates using FedAvg and log round contributors.

        This project uses Quality Inference (QI) as *contribution evaluation only*:
        we log who participated in each round and the centralized global metrics,
        then compute QI scores offline from these logs.
        """
        replies_list = list(replies)

        # Log contributors (src_node_id) for offline QI
        contributors = [msg.metadata.src_node_id for msg in replies_list]

        # Build node_id -> partition-id mapping from per-client metrics
        for msg in replies_list:
            nid = msg.metadata.src_node_id
            try:
                m = msg.content.get("metrics", None)
                if m is not None and "partition-id" in m:
                    pid = int(float(m["partition-id"]))
                    self.log.node_id_map[str(nid)] = pid
            except Exception:
            # If anything unexpected happens, just skip mapping for this msg
                pass



        self.log.contributors[server_round] = contributors

        # Try to learn mapping node_id -> partition-id from client metrics (if provided)
        for msg in replies_list:
            nid = msg.metadata.src_node_id
            pid = None
            try:
                # metrics_agg is aggregated; we need per-client metrics from msg.content
                # Flower puts metrics in the MetricRecord keyed by self.metricrecord_key
                content = msg.content
                if self.metricrecord_key in content:
                    mr = content[self.metricrecord_key]  # MetricRecord
                    if "partition-id" in mr:
                        pid = int(_f(mr["partition-id"]))
            except Exception:
                pid = None

            if pid is not None:
                self.log.node_id_map[str(nid)] = pid

        self.log.dump()

        # Keep for convenience/debugging (not used for weighting)
        self._contributors_by_round[server_round] = contributors
        for nid in contributors:
            self.q.setdefault(nid, 1.0)

        arrays_agg, metrics_agg = super().aggregate_train(server_round, replies_list)

        # Cache global arrays for potential future extensions (not required for offline     QI)
        if arrays_agg is not None:
            self._current_global = arrays_agg



        return arrays_agg, metrics_agg
