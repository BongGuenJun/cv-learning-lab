import csv
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from model import SmallCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "channel_comparison"

# 두 모델에 동일한 CPU 스레드 조건 적용
torch.set_num_threads(1)
torch.manual_seed(42)

# 이미지 내용의 정확도가 아니라 실행시간을 비교하므로 같은 가상 입력 사용
images = torch.randn(1, 3, 32, 32)

models = {}

for name in ["channels_8_16", "channels_4_8"]:
    checkpoint = torch.load(
        RESULTS_DIR / f"{name}.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = SmallCNN(channels=tuple(checkpoint["channels"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    models[name] = model

warmup = 30
calls_per_round = 100
rounds = 10

measurements = {name: [] for name in models}

with torch.no_grad():
    # 초기 실행 준비 비용을 측정에서 분리
    for model in models.values():
        for _ in range(warmup):
            model(images)

    names = list(models.keys())

    for round_index in range(rounds):
        # 항상 한 모델이 먼저 측정되는 것을 피하도록 순서 교대
        order = names if round_index % 2 == 0 else names[::-1]

        for name in order:
            model = models[name]

            start = perf_counter()

            for _ in range(calls_per_round):
                model(images)

            elapsed = perf_counter() - start

            # 초 → 밀리초 변환 후 호출당 평균 계산
            milliseconds = elapsed * 1000 / calls_per_round
            measurements[name].append(milliseconds)

rows = []

for name, values in measurements.items():
    rows.append({
        "model": name,
        "median_ms": median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "batch_size": 1,
        "input_shape": "1x3x32x32",
        "device": "cpu",
        "dtype": "float32",
        "threads": torch.get_num_threads(),
        "warmup": warmup,
        "calls_per_round": calls_per_round,
        "rounds": rounds,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "system": platform.platform(),
        "processor": platform.processor(),
    })

output_path = RESULTS_DIR / "inference_benchmark.csv"

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(
        f"{row['model']} | "
        f"호출당 평균의 중앙값: {row['median_ms']:.4f} ms | "
        f"묶음별 범위: {row['min_ms']:.4f}~"
        f"{row['max_ms']:.4f} ms"
    )

print("측정 결과 저장:", output_path)