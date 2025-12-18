"""GTG-Shapley contribution evaluation wrapper for Flower ServerApp FedAvg.

Important: GTG does NOT change aggregation; we only compute and log contributions
after the normal FedAvg aggregation step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, test
from pytorchexample.torch_shapley import TorchGTGShapley


def _f(x: object, default: float = 0.0) -> float:
    """Safely coerce MetricRecord values (JSON-ish) to float."""
    try:
        return float(x)
    except Exception:
        return default


class GTGFedAvg(FedAvg):
    """FedAvg + GTG-Shapley contribution evaluation (logging only)."""

    def __init__(
        self,
        *,
        fraction_train: float,
        fraction_evaluate: float,
        gtg_out_dir: str = "gtg_logs",
        **kwargs,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            **kwargs,
        )

        self.gtg_out_dir = Path(gtg_out_dir)
        self.gtg_out_dir.mkdir(parents=True, exist_ok=True)

        # Utility baseline for GTG's empty coalition. We'll use "previous round global accuracy".
        self.last_round_utility: float = 0.0

        # Fixed validation loader (same every round)
        self.val_loader = load_centralized_dataset()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def aggregate_train(
        self,
        server_round: int,
        replies: List[Message],
    ) -> tuple[ArrayRecord, MetricRecord] | None:
        """Aggregate as usual, then compute GTG-Shapley on the same replies."""
        aggregated = super().aggregate_train(server_round, replies)
        if aggregated is None:
            return None

        agg_arrays, agg_metrics = aggregated

        # Extract client models + weights from replies (order matters: GTG indices refer to this list)
        client_states = [msg.content["arrays"].to_torch_state_dict() for msg in replies]
        client_weights = [
            _f(msg.content["metrics"].get("num-examples", 1.0), 1.0) for msg in replies
        ]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


       # def eval_state(state: Dict[str, torch.Tensor]) -> float:
        #    """Return central accuracy for a given model state."""
        #    model = Net()
        #    model.load_state_dict(state)
        #    model.to(device)
        #    _, acc = test(model, self.val_loader, device)
        #    return float(acc)
        

        def eval_state(state_dict) -> float:
            model = Net()
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            with torch.no_grad():
                loss, acc = test(model, self.val_loader, self.device)

             # choose ONE:
            return float(loss)   # recommended (stable, consistent with "lower is better")
            # return float(acc)  # only if you also fix best-subset selection logic

        def weighted_average_state(indices: List[int]) -> Dict[str, torch.Tensor]:
            """FedAvg-style weighted average over a subset of client states."""
            # Start from zeros with same shapes
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
                # Fallback to simple mean if weights are degenerate
                n = float(len(indices))
                for k in avg.keys():
                    avg[k] = avg[k] / n
            else:
                for k in avg.keys():
                    avg[k] = avg[k] / total_w

            return avg

        # Utility function for GTG: accuracy of subset-aggregated model
        def utility_fn(subset: List[int]) -> float:
            if len(subset) == 0:
                return float(self.last_round_utility)
            subset_state = weighted_average_state(subset)
            return eval_state(subset_state)

        # Compute current global utility from aggregated model (for logging + baseline update)
        current_global_utility = eval_state(agg_arrays.to_torch_state_dict())

        gtg = TorchGTGShapley(
            num_players=len(replies),
            last_round_utility=float(self.last_round_utility),
            normalize=False,  # keep raw utility contributions; change to True if you want normalized
            device=device,
        )
        gtg.set_utility_function(utility_fn)
        shapley_all, shapley_best = gtg.compute(round_num=server_round)

        out = {
            "round": int(server_round),
            "utility_prev_round": float(self.last_round_utility),
            "utility_current_global": float(current_global_utility),
            "num_clients": int(len(replies)),
            "client_weights_num_examples": [float(w) for w in client_weights],
            "shapley_all": {str(k): float(v) for k, v in shapley_all.items()},
            "shapley_best_subset": {str(k): float(v) for k, v in shapley_best.items()},
        }
        with open(self.gtg_out_dir / f"round_{server_round:03d}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        # Update baseline for next round
        self.last_round_utility = float(current_global_utility)

        return agg_arrays, agg_metrics
