import importlib
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from IPython.core.display import HTML
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from statistics import fmean

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"

root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part3_optimization.tests as tests
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
from plotly_utils import bar, imshow, line

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)


WORLD_SIZE = min(t.cuda.device_count(), 3)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"


def send_receive_nccl(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")

    if rank == 0:
        # Create a tensor, send it to rank 1
        sending_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, sending {sending_tensor=}")
        dist.send(sending_tensor, dst=1)  # Send tensor to CPU before sending
    elif rank == 1:
        # Receive tensor from rank 0 (it needs to be on the CPU before receiving)
        received_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, creating {received_tensor=}")
        dist.recv(
            received_tensor, src=0
        )  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, {device=}, received {received_tensor=}")

    dist.destroy_process_group()


# if MAIN:
#     world_size = 2  # simulate 2 processes
#     mp.spawn(
#         send_receive_nccl,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True,
#     )


def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank 0 to all other ranks.
    """
    if rank == src:
        for dst in range(world_size):
            if dst != src:
                dist.send(tensor, dst)
    else:
        dist.recv(tensor, src)


# if MAIN:
#     tests.test_broadcast(broadcast, WORLD_SIZE)


def reduce(tensor, rank, world_size, dst=0, op: Literal["sum", "mean"] = "sum"):
    """
    Reduces gradients to rank `dst`, so this process contains the sum or mean of all tensors across
    processes.
    """
    if rank == dst:
        tensors = t.stack([t.zeros_like(tensor) for _ in range(world_size)])
        for src in range(world_size):
            if src != dst:
                dist.recv(tensors[src], src)
        if op == "sum":
            op = t.sum
        else:
            op = t.mean
        tensors[dst] = tensor
        tensor.copy_(op(tensors, dim=0))
    else:
        dist.send(dst=dst, tensor=tensor)


def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """
    if rank != 0:
        dist.send(tensor, 0)
        dist.recv(tensor, 0)
    else:
        tensors = t.stack([t.zeros_like(tensor) for _ in range(world_size)])
        tensors[0] = tensor
        for src in range(1, world_size):
            dist.recv(tensors[src], src)
        if op == "sum":
            op = t.sum
        else:
            op = t.mean
        tensor.copy_(op(tensors, dim=0))
        for dst in range(1, world_size):
            dist.send(tensor, dst)


# if MAIN:
#     tests.test_reduce(reduce, WORLD_SIZE)
#     tests.test_all_reduce(all_reduce, WORLD_SIZE)


class SimpleModel(t.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.param = t.nn.Parameter(t.tensor([2.0]))

    def forward(self, x: Tensor):
        return x - self.param


def run_simple_model(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")
    model = SimpleModel().to(device)  # Move the model to the device corresponding to this process
    optimizer = t.optim.SGD(model.parameters(), lr=0.1)

    input = t.tensor([rank], dtype=t.float32, device=device)
    output = model(input)
    loss = output.pow(2).sum()
    loss.backward()  # Each rank has separate gradients at this point

    print(f"Rank {rank}, before all_reduce, grads: {model.param.grad=}")
    all_reduce(model.param.grad, rank, world_size)  # Synchronize gradients
    print(
        f"Rank {rank}, after all_reduce, synced grads (summed over processes): {model.param.grad=}"
    )

    optimizer.step()  # Step with the optimizer (this will update all models the same way)
    print(f"Rank {rank}, new param: {model.param.data}")

    dist.destroy_process_group()


# if MAIN:
#     world_size = 2
#     mp.spawn(
#         run_simple_model,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True,
#     )


def get_untrained_resnet(n_classes: int) -> ResNet34:
    """
    Gets untrained resnet using code from part2_cnns.solutions (you can replace this with your
    implementation).
    """
    resnet = ResNet34()
    resnet.out_layers[-1] = Linear(resnet.out_features_per_group[-1], n_classes)
    return resnet


@dataclass
class ResNetFinetuningArgs:
    n_classes: int = 10
    batch_size: int = 128
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class WandbResNetFinetuningArgs(ResNetFinetuningArgs):
    """Contains new params for use in wandb.init, as well as all the ResNetFinetuningArgs params."""

    wandb_project: str | None = "day3-resnet"
    wandb_name: str | None = None


@dataclass
class DistResNetTrainingArgs(WandbResNetFinetuningArgs):
    world_size: int = 1
    wandb_project: str | None = "day3-resnet-dist-training"


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def get_cifar() -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Returns CIFAR-10 train and test sets."""
    cifar_trainset = datasets.CIFAR10(
        exercises_dir / "data", train=True, download=True, transform=IMAGENET_TRANSFORM
    )
    cifar_testset = datasets.CIFAR10(
        exercises_dir / "data", train=False, download=True, transform=IMAGENET_TRANSFORM
    )
    return cifar_trainset, cifar_testset


class DistResNetTrainer:
    args: DistResNetTrainingArgs

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")

    def pre_training_setup(self):
        self.model = get_untrained_resnet(n_classes=self.args.n_classes).to(self.device)
        if self.rank == 0:
            wandb.init()
        if self.args.world_size > 1:
            for param in self.model.parameters():
                broadcast(rank=self.rank, world_size=self.args.world_size, src=0, tensor=param.data)

        self.trainset, self.testset = get_cifar()
        self.train_sampler = None
        self.test_sampler = None
        if self.args.world_size > 1:
            self.train_sampler = t.utils.data.DistributedSampler(
                self.trainset,
                num_replicas=self.args.world_size,  # we'll divide each batch up into this many random sub-batches
                rank=self.rank,  # this determines which sub-batch this process gets
            )
            self.test_sampler = t.utils.data.DistributedSampler(
                self.testset, num_replicas=self.args.world_size, rank=self.rank
            )
        self.train_loader = t.utils.data.DataLoader(
            self.trainset,
            self.args.batch_size,  # this is the sub-batch size, i.e. the batch size that each GPU gets
            sampler=self.train_sampler,
            num_workers=2,  # setting this low so as not to risk bottlenecking CPU resources
            pin_memory=True,  # this can improve data transfer speed between CPU and GPU
        )
        self.test_loader = t.utils.data.DataLoader(
            self.testset,  # Fixed: was using trainset instead of testset
            self.args.batch_size,
            sampler=self.test_sampler,
            num_workers=2,
            pin_memory=True,
        )

        self.examples_seen = 0

        if self.rank == 0:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_name,
                config=self.args,
            )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()  # Each rank has separate gradients at this point
        if self.args.world_size > 1:
            for param in self.model.parameters():
                all_reduce(
                    param.grad, self.rank, self.args.world_size, op="mean"
                )  # Synchronize gradients

        self.optimizer.step()  # Step with the optimizer (this will update all models the same way)
        self.optimizer.zero_grad()
        self.examples_seen += imgs.shape[0] * self.args.world_size
        if self.rank == 0:
            wandb.log(
                {
                    "train_loss": loss.item(),
                },
                step=self.examples_seen,
            )
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        total_correct = 0
        total_samples = 0
        for imgs, labels in self.test_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            preds = logits.argmax(dim=1)
            total_correct += (labels == preds).sum().item()
            total_samples += imgs.shape[0]
        tensor = t.tensor([total_correct, total_samples], device=self.device)
        all_reduce(tensor=tensor, rank=self.rank, world_size=self.args.world_size, op="sum")
        total_correct, total_samples = tensor.tolist()
        accuracy = total_correct / total_samples
        if self.rank == 0:
            wandb.log({"accuracy": accuracy}, step=self.examples_seen)
        return accuracy

    def train(self):
        self.pre_training_setup()
        for epoch in range(self.args.epochs):
            if self.args.world_size > 1:
                self.train_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)
            for imgs, labels in self.train_loader:
                self.training_step(imgs, labels)
            self.evaluate()
        if self.rank == 0:
            wandb.finish()
            t.save(self.model.state_dict, f"resnet.pth")


def dist_train_resnet_from_scratch(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size)
    trainer = DistResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()


if MAIN:
    wandb.login()
    world_size = t.cuda.device_count()
    mp.spawn(
        dist_train_resnet_from_scratch,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
