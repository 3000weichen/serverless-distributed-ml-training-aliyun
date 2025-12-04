import os
import time
import json
import tarfile

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from flask import Flask, request, jsonify


# -------------------
# Path configuration
# -------------------
TMP_DIR = "/data/tmp"
WEIGHT_DIR = os.path.join(TMP_DIR, "weights")
STATS_DIR = os.path.join(TMP_DIR, "stats")

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

app = Flask(__name__)


# -------------------
# Data preparation
# -------------------
def prepare_cifar10_from_oss():
    """
    Prepare CIFAR-10 dataset from OSS mounted directory into /tmp.

    On Aliyun Function Compute, the OSS bucket is mounted under /data/tmp.
    We copy and extract the dataset into /tmp so that torchvision can use it.
    """
    target_folder = "/tmp/cifar-10-batches-py"
    if os.path.exists(target_folder):
        print("✅ CIFAR-10 dataset already exists in /tmp, skip download.")
        return

    oss_tar_path = "/data/tmp/cifar-10-python.tar.gz"
    tmp_tar_path = "/tmp/cifar-10-python.tar.gz"

    if not os.path.exists(oss_tar_path):
        raise FileNotFoundError(f"❌ CIFAR-10 tar not found at: {oss_tar_path}")

    print("📦 Copying CIFAR-10 from OSS to /tmp ...")
    with open(oss_tar_path, "rb") as src, open(tmp_tar_path, "wb") as dst:
        dst.write(src.read())

    print("📂 Extracting CIFAR-10 ...")
    with tarfile.open(tmp_tar_path, "r:gz") as tar:
        tar.extractall(path="/tmp")

    print("✅ CIFAR-10 prepared under /tmp.")


_transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def evaluate(model: nn.Module, batch_size: int):
    """
    Evaluate the global model on CIFAR-10 test set.
    """
    model.eval()
    testset = CIFAR10(root="/tmp", train=False, download=False, transform=_transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item() * y.size(0)
            _, pred = torch.max(out, dim=1)
            total_correct += (pred == y).sum().item()

    avg_loss = total_loss / len(testset)
    avg_acc = total_correct / len(testset)
    return avg_loss, avg_acc


def average_weights(weight_list):
    """
    Simple FedAvg: average a list of state_dicts.
    """
    avg = {}
    for k in weight_list[0].keys():
        avg[k] = sum(w[k] for w in weight_list) / len(weight_list)
    return avg


# -------------------
# Main coordination endpoint
# -------------------
@app.route("/train", methods=["POST"])
def coordinator():
    """
    Coordinator (parameter server) for serverless distributed training.

    This endpoint:
    - Loads configuration from request JSON
    - Initializes a global ResNet18 model
    - Aggregates worker weights at synchronization epochs
    - Evaluates the model and saves checkpoints/statistics
    """
    request_start = time.time()

    data = request.get_json()
    print("🚀 Starting coordination with config:", data)

    prepare_cifar10_from_oss()

    global_epoch = int(data["epoch"])
    worker_num = int(d_
