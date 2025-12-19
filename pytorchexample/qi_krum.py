# pytorchexample/qi_krum.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy import MultiKrum


def _f(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class QILog:
    out_dir: Path = Path("qi_logs")
    contributors: Dict[int, List[int]] = field(default_factory=dict)  # round -> [node_id]
    acc: List[float] = field(default_factory=list)                   # index = server_round
    loss: List[float] = field(default_factory=list)
    node_id_map: Dict[str, int] = field(default_factory=dict)        # str(node_id) -> partition-id

    def dump(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "contributors.json").write_text(
            json.dumps(self.contributors, indent=2, sort_keys=True), encoding="utf-8"
        )
        (self.out_dir / "global_metrics.json").write_text(
            json.dumps({"accuracy": self.acc, "loss": self.loss}, indent=2), encoding="utf-8"
        )
        (self.out_dir / "node_id_map.json").write_text(
            json.dumps(self.node_id_map, indent=2, sort_keys=True), encoding="utf-8"
        )


class QIMultiKrum(MultiKrum):
    """MultiKrum + QI logging (contributors + centralized metrics), for offline_qi.py."""

    def __init__(self, *, qi_out_dir: str = "qi_logs", **kwargs) -> None:
        super().__init__(**kwargs)
        self.log = QILog(out_dir=Path(qi_out_dir))

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
        # Wrap evaluate_fn to log global metrics per round (same as QIFedAvg.start)
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

            while len(self.log.acc) <= server_round:
                self.log.acc.append(0.0)
                self.log.loss.append(0.0)
            self.log.acc[server_round] = acc
            self.log.loss[server_round] = loss
            self.log.dump()
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

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        replies_list = list(replies)

        # Log contributors (Flower node_id)
        contributors = [msg.metadata.src_node_id for msg in replies_list]
        self.log.contributors[server_round] = contributors

        # Log node_id -> partition-id mapping (from client metrics)
        for msg in replies_list:
            nid = msg.metadata.src_node_id
            try:
                mr = msg.content.get("metrics", None)
                if mr is not None and "partition-id" in mr:
                    pid = int(_f(mr["partition-id"]))
                    self.log.node_id_map[str(nid)] = pid
            except Exception:
                pass

        self.log.dump()

        # Run MultiKrum aggregation as normal
        return super().aggregate_train(server_round, replies_list)
