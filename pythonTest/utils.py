import subprocess
import torch

def pick_best_device(min_free_gb=10.0):
    if not torch.cuda.is_available():
        return torch.device("cpu")

    query = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]

    try:
        output = subprocess.check_output(query, encoding="utf-8")
    except Exception as exc:
        print(f"Could not query nvidia-smi: {exc}")
        return torch.device("cpu")

    best_gpu = None
    best_score = None

    for line in output.strip().splitlines():
        gpu_id, gpu_util, free_mem, total_mem = [x.strip() for x in line.split(",")]
        gpu_id = int(gpu_id)
        gpu_util = float(gpu_util)
        free_mem = float(free_mem) / 1024.0  # MB -> GB
        total_mem = float(total_mem) / 1024.0

        print(f"GPU {gpu_id}: util={gpu_util:.0f}% free={free_mem:.2f} GB / total={total_mem:.2f} GB")

        if free_mem < min_free_gb:
            continue

        score = (gpu_util, -free_mem)
        if best_score is None or score < best_score:
            best_score = score
            best_gpu = gpu_id

    if best_gpu is None:
        print("No GPU meets the free-memory threshold. Falling back to CPU.")
        return torch.device("cpu")

    print(f"Selecting GPU {best_gpu}")
    return torch.device(f"cuda:{best_gpu}")