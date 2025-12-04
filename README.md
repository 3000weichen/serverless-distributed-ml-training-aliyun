# Serverless Distributed Machine Learning on AliCloud

**Optimizing Parameter Configuration for Efficient Serverless ML Training**  
*(Based on ResNet-18 + CIFAR-10 on AliCloud Function Compute)*

This project implements a fully serverless distributed training system and performs a **systematic parameter optimization study** to enable **faster, cheaper, and scalable** machine learning on pay-per-use cloud platforms.

This repository is based on my research work:  
“**Optimizing Parameter Configuration for Efficient Serverless Distributed Machine Learning on AliCloud**” :contentReference[oaicite:1]{index=1}

---

## Motivation

Traditional ML clusters require long-running servers → costly and complex  
Serverless → auto-scaling, pay-per-use ⚡ but ✘ high communication and cold-start costs

 **Research Goal**  
Find the optimal configuration of:

- `batch_size`
- `local_epoch`
- `sync_freq`

to **minimize time and resource usage while retaining accuracy**.

---

##  Architecture

Server–Worker collaborative training using AliCloud Function Compute + OSS
<img width="1093" height="435" alt="image" src="https://github.com/user-attachments/assets/67ab445a-ffeb-4ccb-acaa-3f1d856b97a3" />


(Architecture figure adapted from my paper, Figure 1) :contentReference[oaicite:2]{index=2}

---

## Model & Dataset

| Component | Description |
|---|---|
| Model | ResNet-18 (SGD: lr=0.01, momentum=0.9) |
| Dataset | CIFAR-10 (OSS-hosted, auto-extracted) |
| Workers | 10 parallel functions (each with data shard) |

---

## Repository Structure
src/
server/handler.py # Parameter server: aggregation & evaluation
worker/handler.py # Worker function: shard training + sync
orchestrator/ # Local launcher for server & workers

experiments/
batch_size_32/ # Experiment logs for different configurations
batch_size_64/
batch_size_128/
local_epoch_1/
local_epoch_2/
local_epoch_5/
sync_freq_1/
sync_freq_2/
sync_freq_3/


## Key Research Results (from the paper)

Three core parameters were evaluated individually and jointly :contentReference[oaicite:1]{index=1}：

| Parameter | Best Setting | Reason |
|---|:---:|---|
| Batch Size | **128** | ↑ accuracy **AND** ↓ training time |
| Local Epoch | **2** | Best accuracy/time trade-off |
| Sync Frequency | **2** | Balance convergence vs OSS I/O cost |

---

### Final Optimal Configuration

| Metric | Result |
|---|:---:|
| Accuracy | **0.7052** (97.1% of peak) |
| Total Training Time | **265.32 sec** ⏱ |
| Storage Traffic | **↓ 50%** |
| Function Invocations | **↓ 63%** |
| Cold-starts | **↓ 40%** |

This is **129.4% faster and 89.6% cheaper**  
than poorly-tuned configurations with high local epochs :contentReference[oaicite:2]{index=2}

---

## What I Learned (and demonstrated)

✔ Serverless distributed system engineering  
✔ Performance tuning & cloud-cost efficiency  
✔ Experiment design & ML systems evaluation  
✔ Full-stack pipeline: deployment → logs → plots → paper writing  

> This project proves that **serverless ML is production-feasible** with proper parameter optimization.

---

## Future Work

- Adaptive parameter scheduling (e.g., RL-based tuning)
- Communication compression (reduce OSS I/O)
- Cross-cloud benchmarking (AWS / GCP)

---

## Author

**Hao Chen**   
📫 h.chen.14@umail.leidenuniv.nl


