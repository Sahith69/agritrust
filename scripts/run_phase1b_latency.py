import sys
sys.path.insert(0, '/home/sahith/Projects/Crop')
import config
import os
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.model import CropSeverityModel
from src.dataset import CropDiseaseDataset, get_transforms
from src.corn_loss import corn_probs_to_grade

start_time = time.time()
os.makedirs('/home/sahith/Projects/Crop/outputs/improvements', exist_ok=True)
device = 'cpu'

# 1. Training time check
training_time_notes = {
    "logged_training_time_available": False,
    "reason": "Model checkpoints (.pth files) store exclusively OrderedDict state_dicts (weights/biases) without epoch timestamps. Training scripts did not log cumulative wall-clock timers to disk. Total training time could not be extracted from historical artifacts."
}

# 2. Benchmark inference latency on 50 test images per model
models_to_test = [
    {
        "name": "Paddy ResNet-18 (CORN)",
        "backbone": "resnet18",
        "ordinal": True,
        "path": os.path.join(config.MODELS_DIR, "paddy_ordinal.pth"),
        "split_csv": config.PADDY_SPLIT_CSV
    },
    {
        "name": "Paddy ResNet-18 (Softmax)",
        "backbone": "resnet18",
        "ordinal": False,
        "path": os.path.join(config.MODELS_DIR, "paddy_baseline.pth"),
        "split_csv": config.PADDY_SPLIT_CSV
    },
    {
        "name": "Arecanut ResNet-18 (CORN)",
        "backbone": "resnet18",
        "ordinal": True,
        "path": os.path.join(config.MODELS_DIR, "arecanut_ordinal.pth"),
        "split_csv": config.ARECANUT_SPLIT_CSV
    },
    {
        "name": "Arecanut ResNet-18 (Softmax)",
        "backbone": "resnet18",
        "ordinal": False,
        "path": os.path.join(config.MODELS_DIR, "arecanut_baseline.pth"),
        "split_csv": config.ARECANUT_SPLIT_CSV
    },
    {
        "name": "Arecanut ResNet-34 (CORN)",
        "backbone": "resnet34",
        "ordinal": True,
        "path": os.path.join(config.MODELS_DIR, "architecture_comparison/arecanut_resnet34_ordinal.pth"),
        "split_csv": config.ARECANUT_SPLIT_CSV
    },
    {
        "name": "Arecanut MobileNetV2 (CORN)",
        "backbone": "mobilenet_v2",
        "ordinal": True,
        "path": os.path.join(config.MODELS_DIR, "architecture_comparison/arecanut_mobilenet_v2_ordinal.pth"),
        "split_csv": config.ARECANUT_SPLIT_CSV
    }
]

latency_results = {}
NUM_BENCHMARK_IMAGES = 50
NUM_WARMUP = 5

for m_info in models_to_test:
    mname = m_info["name"]
    print(f"Benchmarking latency for {mname}...")
    
    model = CropSeverityModel(backbone=m_info["backbone"], num_classes=config.NUM_SEVERITY_GRADES, ordinal=m_info["ordinal"], pretrained=False)
    model.load_state_dict(torch.load(m_info["path"], map_location=device, weights_only=True))
    model.eval()
    
    ds = CropDiseaseDataset(m_info["split_csv"], 'test', get_transforms('test'))
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    # Warmup
    it = iter(loader)
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            img = next(it)[0]
            _ = model(img)
            
    # Measure
    latencies_ms = []
    with torch.no_grad():
        for _ in range(NUM_BENCHMARK_IMAGES):
            img = next(it)[0]
            t0 = time.perf_counter()
            out = model(img)
            if m_info["ordinal"]:
                _ = corn_probs_to_grade(torch.sigmoid(out))
            else:
                _ = torch.argmax(torch.softmax(out, dim=1), dim=1)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            
    latencies_ms = np.array(latencies_ms)
    latency_results[mname] = {
        "device": "CPU (Intel Core i5-13420H)",
        "batch_size": 1,
        "num_images_benchmarked": NUM_BENCHMARK_IMAGES,
        "latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "std": float(np.std(latencies_ms)),
            "median": float(np.median(latencies_ms)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "min": float(np.min(latencies_ms)),
            "max": float(np.max(latencies_ms))
        }
    }
    print(f"  Mean: {np.mean(latencies_ms):.2f} ms | Median: {np.median(latencies_ms):.2f} ms | p95: {np.percentile(latencies_ms, 95):.2f} ms")

output_data = {
    "training_time": training_time_notes,
    "inference_latency": latency_results
}

out_path = '/home/sahith/Projects/Crop/outputs/improvements/latency_benchmark.json'
with open(out_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Saved latency benchmark to {out_path}")
print(f"Total time: {time.time() - start_time:.2f} seconds")
