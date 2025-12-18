"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, MultiKrum
from pytorchexample.qi_fedavg import QIFedAvg

from pytorchexample.task import Net, load_centralized_dataset, test


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
   # strategy = FedAvg(fraction_evaluate=fraction_evaluate)
   # strategy = QIFedAvg(
   # fraction_train=context.run_config.get("fraction-train", 0.5),
   # fraction_evaluate=fraction_evaluate,
   # qi_out_dir=context.run_config.get("qi-out-dir", "qi_logs"),
#)

    strategy_name = str(context.run_config.get("strategy", "fedavg")).lower()

    if strategy_name in ["qi", "qi-fedavg", "qifedavg"]:
        strategy = QIFedAvg(
            fraction_train=context.run_config.get("fraction-train", 0.5),
            fraction_evaluate=fraction_evaluate,
            qi_out_dir=context.run_config.get("qi-out-dir", "qi_logs"),
        )

    elif strategy_name in ["krum", "multikrum"]:
        # IMPORTANT: Krum needs enough clients *per round*.
        # With 5 total clients and 1 attacker, sample ALL 5 each round.
        num_mal = int(context.run_config.get("num-malicious-nodes", 1))
        strategy = MultiKrum(
            fraction_train=1.0,
            min_train_nodes=context.run_config.get("min-train-nodes", 5),
            min_available_nodes=context.run_config.get("min-available-nodes", 5),
            fraction_evaluate=fraction_evaluate,
            min_evaluate_nodes=context.run_config.get("min-evaluate-nodes", 0),
            num_malicious_nodes=num_mal,
            num_nodes_to_select=1,  # => classical Krum :contentReference[oaicite:1]{index=1}
        )

    else:
        strategy = FedAvg(
            fraction_train=context.run_config.get("fraction-train", 0.5),
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=context.run_config.get("min-train-nodes", 2),
            min_available_nodes=context.run_config.get("min-available-nodes", 2),
            min_evaluate_nodes=context.run_config.get("min-evaluate-nodes", 0),
        )


    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
