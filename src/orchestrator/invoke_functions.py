import os
import threading
import requests
import argparse


def invoke_server(server_url, epoch, batch_size, local_epoch, worker_num, sync_freq, checkpoint_policy):
    """Invoke the parameter server function."""
    data = {
        "epoch": epoch,
        "batch_size": batch_size,
        "local_epoch": local_epoch,
        "worker_num": worker_num,
        "sync_freq": sync_freq,
        "checkpoint_policy": checkpoint_policy,
    }
    response = requests.post(server_url, json=data)
    if response.status_code == 200:
        print("✅ Server invoked successfully.")
    else:
        print(f"❌ Failed to invoke server: {response.status_code}, body={response.text}")


def invoke_worker(worker_url, worker_id, epoch, batch_size, local_epoch, worker_num, sync_freq):
    """Invoke one worker function."""
    data = {
        "worker_id": worker_id,
        "epoch": epoch,
        "batch_size": batch_size,
        "local_epoch": local_epoch,
        "worker_num": worker_num,
        "sync_freq": sync_freq,
    }
    response = requests.post(worker_url, json=data)
    if response.status_code == 200:
        print(f"✅ Worker {worker_id} invoked successfully.")
    else:
        print(f"❌ Failed to invoke worker {worker_id}: {response.status_code}, body={response.text}")


def main(args):
    # 从环境变量或命令行获取 URL
    server_url = args.server_url or os.environ.get("SERVER_URL")
    worker_url = args.worker_url or os.environ.get("WORKER_URL")

    if not server_url or not worker_url:
        raise ValueError("SERVER_URL and WORKER_URL must be provided via args or environment variables.")

    threads = []

    # 先触发 server
    t = threading.Thread(
        target=invoke_server,
        args=(
            server_url,
            args.epoch,
            args.batch_size,
            args.local_epoch,
            args.worker_num,
            args.sync_freq,
            args.checkpoint_policy,
        ),
    )
    threads.append(t)

    # 再触发多个 worker
    for i in range(args.worker_num):
        t = threading.Thread(
            target=invoke_worker,
            args=(
                worker_url,
                i,
                args.epoch,
                args.batch_size,
                args.local_epoch,
                args.worker_num,
                args.sync_freq,
            ),
        )
        threads.append(t)

    # 启动所有线程
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("🎉 All invocations finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator for serverless distributed training on Aliyun.")
    parser.add_argument("--server-url", type=str, default=None, help="Endpoint of the server function.")
    parser.add_argument("--worker-url", type=str, default=None, help="Endpoint of the worker function.")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epoch", type=int, default=5)
    parser.add_argument("--worker-num", type=int, default=10)
    parser.add_argument("--sync-freq", type=int, default=1)
    parser.add_argument("--checkpoint-policy", type=str, default="all", choices=["all", "last", "none"])

    args = parser.parse_args()
    main(args)
