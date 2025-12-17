"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    """
    attacker_enabled_global = bool(context.run_config.get("attacker-enabled", False))
    attacker_id = int(context.run_config.get("attacker-id", -1))  # e.g. 0
    flip_prob = float(context.run_config.get("flip-prob", 0.2))    # 0.2 = 20% label flip
    attack_seed = int(context.run_config.get("attack-seed", 123))

    is_attacker = (partition_id == attacker_id)
    attack_enabled = attacker_enabled_global and is_attacker
    """

    partition_id = context.node_config["partition-id"]

    attacker_enabled = bool(context.run_config.get("attacker-enabled", False))
    attacker_id = int(context.run_config.get("attacker-id", -1))
    flip_prob = float(context.run_config.get("flip-prob", 0.2))
    attack_seed = int(context.run_config.get("attack-seed", 123))

    attack_enabled = attacker_enabled and (partition_id == attacker_id)

    print(f"[client {partition_id}] attack_enabled={attack_enabled}, flip_prob={flip_prob}")


    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
   # trainloader, _ = load_data(partition_id, num_partitions, batch_size)
    trainloader, _ = load_data(
        partition_id,
        num_partitions,
        batch_size,
        non_iid=context.run_config.get("non-iid", False),
        dirichlet_alpha=context.run_config.get("dirichlet-alpha", 0.5),
        seed=context.run_config.get("partition-seed", 42),
    )

    # Call the training function 
    """
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    ) 
    """

    train_loss = train_fn(
    model,
    trainloader,
    context.run_config["local-epochs"],
    msg.content["config"]["lr"],
    device,
    attack_enabled=attack_enabled,
    flip_prob=flip_prob,
    seed=attack_seed,
    client_id=partition_id,
)


    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "partition-id": partition_id,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    # _, valloader = load_data(partition_id, num_partitions, batch_size)
    _, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size,
        non_iid=context.run_config.get("non-iid", False),
        dirichlet_alpha=context.run_config.get("dirichlet-alpha", 0.5),
        seed=context.run_config.get("partition-seed", 42),
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
