import os
import time
import json
import tarfile

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

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


# Data augmentation for local training
_transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def get_local_dataset(worker_id: int, worker_num: int):
    """
    Split CIFAR-10 training set into non-overlapping shards.
    Each worker gets a disjoint subset.
    """
    full_dataset = CIFAR10(
        root="/tmp", train=True, download=False, transform=_transform_train
    )
    total_size = len(full_dataset)
    shard_size = total_size // worker_num

    start = worker_id * shard_size
    # last worker takes the remainder as well
    end = (worker_id + 1) * shard_size if worker_id != worker_num - 1 else total_size

    indices = list(range(start, end))
    print(
        f"Worker {worker_id} will train on samples [{start}, {end}) "
        f"({len(indices)} samples)."
    )
    return Subset(full_dataset, indices)


def train_local_round(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    dataset,
    local_epochs: int,
    batch_size: int,
    device: torch.device,
):
    """
    Perform local training for `local_epochs` on the given dataset.

    This corresponds to one "round" of local training between synchronizations.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model.train()
    for _ in range(local_epochs):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


# -------------------
# Worker training endpoint
# -------------------
@app.route("/train", methods=["POST"])
def handle_train():
    """
    Worker endpoint for local training on a data shard.

    For each epoch:
    - Optionally load the latest global model from the server (synchronization)
    - Perform local training for `local_epoch` epochs
    - Save local weights at synchronization epochs
    """
    start_time = time.time()
    worker_id = None  # in case parsing fails

    try:
        data = request.get_json()
        print("📨 Received worker task:", data)

        worker_id = int(data["worker_id"])
        worker_num = int(data["worker_num"])
        epoch_num = int(data["epoch"])
        batch_size = int(data.get("batch_size", 32))
        local_epoch = int(data.get("local_epoch", 1))
        sync_freq = int(data.get("sync_freq", 1))

        # Prepare dataset
        prepare_cifar10_from_oss()
        local_dataset = get_local_dataset(worker_id, worker_num)

        # Initialize model & optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = resnet18(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        worker_stats = {
            "worker_id": worker_id,
            "epoch_stats": [],
        }

        total_model_size_MB = 0.0

        for epoch in range(epoch_num):
            epoch_start = time.time()

            # Sync epochs: same pattern as server
            is_sync_epoch = (epoch % sync_freq == 0) or (epoch == epoch_num - 1)
            print(
                f"🚀 Worker {worker_id} starts epoch {epoch}/{epoch_num}, "
                f"sync_epoch={is_sync_epoch}"
            )

            # Before training, try to load the latest global weights from server
            # Here we use server_{epoch-1}.pt as the latest checkpoint.
            if epoch > 0:
                server_epoch = epoch - 1
                server_weight_path = os.path.join(
                    WEIGHT_DIR, f"server_{server_epoch}.pt"
                )

                wait_start = time.time()
                timeout_seconds = 600  # 10 min timeout
                retry_count = 0

                print(
                    f"Worker {worker_id} waiting for server weights: "
                    f"{os.path.basename(server_weight_path)}"
                )

                while not os.path.exists(server_weight_path):
                    if time.time() - wait_start > timeout_seconds:
                        raise TimeoutError(
                            f"Timeout waiting for server weights "
                            f"(epoch={server_epoch})"
                        )

                    sleep_time = min(2 ** retry_count, 10)
                    time.sleep(sleep_time)
                    retry_count += 1

                # basic integrity check
                while (
                    os.path.exists(server_weight_path)
                    and os.path.getsize(server_weight_path) < 1024
                ):
                    time.sleep(1)

                state_dict = torch.load(server_weight_path, map_location=device)
                model.load_state_dict(state_dict)
                print(
                    f"🔄 Worker {worker_id} loaded global weights from "
                    f"{os.path.basename(server_weight_path)}"
                )

            # Local training for this epoch
            model = train_local_round(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                dataset=local_dataset,
                local_epochs=local_epoch,
                batch_size=batch_size,
                device=device,
            )

            # Upload local weights only at synchronization epochs
            if is_sync_epoch:
                worker_weight_path = os.path.join(
                    WEIGHT_DIR, f"worker_{worker_id}_epoch_{epoch}.pt"
                )
                torch.save(model.state_dict(), worker_weight_path)
                file_size_MB = os.path.getsize(worker_weight_path) / (1024 * 1024)
                total_model_size_MB += file_size_MB

                print(
                    f"💾 Worker {worker_id} saved local weights: "
                    f"{os.path.basename(worker_weight_path)} "
                    f"({file_size_MB:.2f} MB)"
                )

            epoch_duration = time.time() - epoch_start
            worker_stats["epoch_stats"].append(
                {
                    "epoch": epoch,
                    "is_sync": is_sync_epoch,
                    "duration_sec": epoch_duration,
                }
            )
            print(
                f"⏱️ Worker {worker_id} finished epoch {epoch}, "
                f"duration={epoch_duration:.2f}s"
            )

        # Save worker statistics
        stats_path = os.path.join(STATS_DIR, f"worker_{worker_id}_stats.json")
        try:
            with open(stats_path, "w") as f:
                json.dump(worker_stats, f, indent=2)
            print(f"✅ Worker {worker_id} stats saved to {stats_path}")
        except Exception as e:
            print(f"❌ Failed to save stats for worker {worker_id}: {str(e)}")
            stats_path = None

        return jsonify(
            {
                "status": "success",
                "worker_id": worker_id,
                "duration_sec": time.time() - start_time,
                "train_model_size_MB": round(total_model_size_MB, 2),
                "stats_path": stats_path,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify(
            {
                "status": "error",
                "worker_id": worker_id,
                "error": str(e),
            }
        ), 500


if __name__ == "__main__":
    # For local debugging only
    app.run(host="0.0.0.0", port=9000)
