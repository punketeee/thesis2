"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from collections import Counter


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset
fds_cfg = None  # Cache config used to build FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

_printed_partition_stats = set()

def _print_label_stats_once(partition_id: int, ds, *, prefix: str = "") -> None:
    """Print label histogram once per partition_id per process."""
    key = (prefix, int(partition_id))
    if key in _printed_partition_stats:
        return
    _printed_partition_stats.add(key)

    # HuggingFace Dataset: labels live under "label"
    labels = ds["label"]  # list of ints
    c = Counter(labels)
    total = len(labels)

    # Print sorted by label id for readability
    items = ", ".join([f"{k}:{c.get(k,0)}" for k in sorted(c.keys())])
    print(f"{prefix}[partition {partition_id}] n={total} label_counts={{ {items} }}")


# def load_data(partition_id: int, num_partitions: int, batch_size: int):
def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    non_iid: bool = False,
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    
    #global fds
    #if fds is None:
    #    partitioner = IidPartitioner(num_partitions=num_partitions)
    #    fds = FederatedDataset(
    #        dataset="uoft-cs/cifar10",
    #        partitioners={"train": partitioner},
    #    )

    global fds, fds_cfg

    cfg = (num_partitions, bool(non_iid), float(dirichlet_alpha), int(seed))
    if fds is None or fds_cfg != cfg:
        if non_iid:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=float(dirichlet_alpha),
                seed=int(seed),
            )
        else:
            partitioner = IidPartitioner(num_partitions=num_partitions)

        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
        fds_cfg = cfg


        
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)

    _print_label_stats_once(partition_id, partition_train_test["train"], prefix="[DATA] ")
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)

""""
def train(net, trainloader, epochs, lr, device):
    Train the model on the training set.
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss
"""

"""def train(
    net,
    trainloader,
    epochs,
    lr,
    device,
    *,
    attack_enabled: bool = False,
    flip_prob: float = 0.2,
    num_classes: int = 10,
    seed: int = 42,
    client_id: int = 0,
): """

def train(net, trainloader, epochs, lr, device,
          attack_enabled=False, flip_prob=0.2, num_classes=10, seed=123, client_id=0):

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()

    # deterministic RNG per-client
    gen = torch.Generator()
    gen.manual_seed(int(seed) + 1000 * int(client_id))

    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            # --- attacker: 20% label flip (train only) ---
            if attack_enabled and flip_prob > 0.0:
                mask = torch.rand(labels.shape, generator=gen, device=labels.device) < flip_prob
                if mask.any():
                    # sample a non-zero offset in [1..num_classes-1] so label always changes
                    offsets = torch.randint(
                        low=1, high=num_classes, size=(int(mask.sum().item()),),
                        generator=gen, device=labels.device
                    )
                    labels = labels.clone()
                    labels[mask] = (labels[mask] + offsets) % num_classes
            # ---------------------------------------------

            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # (minor fix) average over all steps across all epochs
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss





def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
