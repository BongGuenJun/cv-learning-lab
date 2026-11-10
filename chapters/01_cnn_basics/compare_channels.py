import csv
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import SmallCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "channel_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 100
BATCH_SIZE = 32

# 두 모델에 같은 데이터와 전처리 사용
dataset = datasets.CIFAR10(
    root=str(PROJECT_ROOT / "data"),
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

split_generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(len(dataset), generator=split_generator)

train_dataset = Subset(dataset, indices[:2000].tolist())
val_dataset = Subset(dataset, indices[2000:3000].tolist())

summary = []

for channels in [(8, 16), (4, 8)]:
    # 모델마다 난수 시작 상태를 다시 설정
    torch.manual_seed(SEED)

    # 별도 생성기로 두 모델의 배치 순서를 동일하게 구성
    loader_generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 모델별로 새 모델과 optimizer 생성
    model = SmallCNN(channels=channels)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    run_name = f"channels_{channels[0]}_{channels[1]}"
    history = []
    run_start = perf_counter()

    print(f"\n실험: {run_name} | 파라미터: {parameter_count:,}개")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            train_samples += batch_size

        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                val_loss_sum += loss.item() * batch_size
                val_correct += (
                    logits.argmax(dim=1) == labels
                ).sum().item()
                val_samples += batch_size

        record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_samples,
            "val_loss": val_loss_sum / val_samples,
            "train_accuracy": train_correct / train_samples,
            "val_accuracy": val_correct / val_samples,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss {record['train_loss']:.4f} / "
            f"{record['val_loss']:.4f} | "
            f"accuracy {record['train_accuracy']:.2%} / "
            f"{record['val_accuracy']:.2%}"
        )

    elapsed = perf_counter() - run_start

    # 모델별 epoch 기록을 서로 다른 파일에 저장
    history_path = RESULTS_DIR / f"{run_name}_history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    # 학습한 모델은 다음 추론시간 측정 단계에서 재사용
    checkpoint_path = RESULTS_DIR / f"{run_name}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "channels": list(channels),
            "classes": dataset.classes,
            "input_size": [32, 32],
            "color_mode": "RGB",
            "preprocessing": "ToTensor",
            "epochs": EPOCHS,
            "seed": SEED,
        },
        checkpoint_path,
    )

    summary.append({
        "model": run_name,
        "parameters": parameter_count,
        "final_train_loss": record["train_loss"],
        "final_val_loss": record["val_loss"],
        "final_train_accuracy": record["train_accuracy"],
        "final_val_accuracy": record["val_accuracy"],
        "train_and_validation_seconds": elapsed,
    })

# 두 모델의 마지막 epoch 결과를 한 표로 저장
summary_path = RESULTS_DIR / "summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(summary[0].keys()),
    )
    writer.writeheader()
    writer.writerows(summary)

print("\n비교 결과")
for row in summary:
    print(
        f"{row['model']} | "
        f"파라미터 {row['parameters']:,}개 | "
        f"검증 accuracy {row['final_val_accuracy']:.2%} | "
        f"학습·검증 시간 "
        f"{row['train_and_validation_seconds']:.2f}초"
    )

print("결과 저장:", RESULTS_DIR)