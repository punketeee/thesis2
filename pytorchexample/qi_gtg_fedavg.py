# pytorchexample/qi_gtg_fedavg.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, test
from pytorchexample.torch_shapley import TorchGTGShapley


def _f(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class QILog:
    out_dir: Path
    contributors: Dict[int, List[int]] = field(default_factory=dict)  # round -> [node_id]
    acc: List[float] = field(default_factory=list)                   # index = server_round
    loss: List[float] = field(default_factory=list)
    node_id_map: Dict[str, int] = field(default_factory=dict)        # str(node_id) -> partition-id

    def dump(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "contributors.json").write_text(
            json.dumps(self.contributors, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.out_dir / "global_metrics.json").write_text(
            json.dumps({"accuracy": self.acc, "loss": self.loss}, indent=2),
            encoding="utf-8",
        )
        (self.out_dir / "node_id_map.json").write_text(
            json.dumps(self.node_id_map, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class QIAndGTGFedAvg(FedAvg):
    """
    FedAvg (normal) + logs for OFFLINE QI + logs for GTG-Shapley.
    No weighting, no robust aggregation changes.
    """

    def __init__(
        self,
        *,
        fraction_train: float,
        fraction_evaluate: float,
        qi_out_dir: str = "qi_logs",
        gtg_out_dir: str = "gtg_logs",
        **kwargs,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            **kwargs,
        )

        # ---- QI logging state ----
        self.qi = QILog(out_dir=Path(qi_out_dir))

        # ---- GTG logging state ----
        self.gtg_out_dir = Path(gtg_out_dir)
        self.gtg_out_dir.mkdir(parents=True, exist_ok=True)
        self.last_round_utility: float = 0.0

        self.val_loader = load_centralized_dataset()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Log global metrics (QI) via wrapped evaluate_fn
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

            while len(self.qi.acc) <= server_round:
                self.qi.acc.append(0.0)
                self.qi.loss.append(0.0)
            self.qi.acc[server_round] = acc
            self.qi.loss[server_round] = loss
            self.qi.dump()

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

    # Aggregate normally, then compute GTG, and log QI contributors
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        replies_list = list(replies)

        # ---------- QI: log contributors + node_id map ----------
        contributors = [msg.metadata.src_node_id for msg in replies_list]
        self.qi.contributors[server_round] = contributors

        for msg in replies_list:
            nid = msg.metadata.src_node_id
            try:
                m = msg.content.get("metrics", None)
                if m is not None and "partition-id" in m:
                    self.qi.node_id_map[str(nid)] = int(float(m["partition-id"]))
            except Exception:
                pass

        self.qi.dump()

        # ---------- FedAvg aggregation (unchanged) ----------
        aggregated = super().aggregate_train(server_round, replies_list)
        if aggregated is None:
            return None, None
        agg_arrays, agg_metrics = aggregated

        # ---------- GTG: compute shapley on the same replies ----------
        client_states = [msg.content["arrays"].to_torch_state_dict() for msg in replies_list]
        client_weights = [
            _f(msg.content["metrics"].get("num-examples", 1.0), 1.0) for msg in replies_list
        ]

        def eval_state(state_dict) -> float:
            model = Net()
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            with torch.no_grad():
                loss, acc = test(model, self.val_loader, self.device)
            return float(loss)  # loss => lower is better (matches your GTG best-subset logic)

        def weighted_average_state(indices: List[int]) -> Dict[str, torch.Tensor]:
            first = client_states[indices[0]]
            avg: Dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in first.items()}

            total_w = 0.0
            for i in indices:
                w = float(client_weights[i])
                total_w += w
                st = client_states[i]
                for k in avg.keys():
                    avg[k] += st[k] * w

            if total_w <= 0.0:
                n = float(len(indices))
                for k in avg.keys():
                    avg[k] = avg[k] / n
            else:
                for k in avg.keys():
                    avg[k] = avg[k] / total_w

            return avg

        def utility_fn(subset: List[int]) -> float:
            if len(subset) == 0:
                return float(self.last_round_utility)
            subset_state = weighted_average_state(subset)
            return eval_state(subset_state)

        current_global_utility = eval_state(agg_arrays.to_torch_state_dict())

        gtg = TorchGTGShapley(
            num_players=len(replies_list),
            last_round_utility=float(self.last_round_utility),
            normalize=False,
            device=self.device,
        )
        gtg.set_utility_function(utility_fn)
        shapley_all, shapley_best = gtg.compute(round_num=server_round)

        out = {
            "round": int(server_round),
            "utility_prev_round": float(self.last_round_utility),
            "utility_current_global": float(current_global_utility),
            "num_clients": int(len(replies_list)),
            "client_weights_num_examples": [float(w) for w in client_weights],
            "shapley_all": {str(k): float(v) for k, v in shapley_all.items()},
            "shapley_best_subset": {str(k): float(v) for k, v in shapley_best.items()},
        }
        (self.gtg_out_dir / f"round_{server_round:03d}.json").write_text(
            json.dumps(out, indent=2),
            encoding="utf-8",
        )

        self.last_round_utility = float(current_global_utility)

        return agg_arrays, agg_metrics
