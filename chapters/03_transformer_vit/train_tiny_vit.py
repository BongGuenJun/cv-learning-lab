import csv
from pathlib import Path
from time import perf_counter

import matplotlib

# 그래프 창을 열지 않고 이미지 파일로 저장
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from tiny_vit import TinyViT


def main():
    project_root = Path(__file__).resolve().parents[2]

    seed = 42
    epochs = 10

    result_dir = project_root / "results" / "tiny_vit"
    checkpoint_dir = project_root / "checkpoints"

    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 이미지를 텐서로 변환하고 픽셀값을 0~1 범위로 변환
    preprocess = transforms.ToTensor()

    dataset = datasets.CIFAR10(
        root=str(project_root / "data"),
        train=True,
        download=True,
        transform=preprocess,
    )

    # 학습용 이미지와 검증용 이미지가 겹치지 않도록 분할
    split_generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(dataset),
        generator=split_generator,
    )

    train_dataset = Subset(dataset, indices[:2000])
    val_dataset = Subset(dataset, indices[2000:3000])

    # 학습 데이터를 섞는 순서를 재현할 수 있도록 설정
    loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # 모델의 초기 가중치를 재현할 수 있도록 시드 설정
    torch.manual_seed(seed)

    model = TinyViT()

    criterion = nn.CrossEntropyLoss()

    # 모든 학습 가능한 파라미터를 AdamW로 업데이트
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
    )

    print(f"학습 이미지 수: {len(train_dataset)}")
    print(f"검증 이미지 수: {len(val_dataset)}")
    print(
        "파라미터 수:",
        sum(parameter.numel() for parameter in model.parameters()),
    )

    history = []

    for epoch in range(1, epochs + 1):
        start = perf_counter()

        # 학습 모드 설정
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            # 이전 배치에서 계산한 기울기 초기화
            optimizer.zero_grad()

            # 이미지로 클래스별 점수 계산
            logits = model(images)

            # 예측 점수와 정답 클래스 번호로 손실 계산
            loss = criterion(logits, labels)

            # 기울기 계산 후 파라미터 업데이트
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            # 배치 평균 손실을 이미지 수 기준으로 누적
            train_loss_sum += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_samples += batch_size

        train_loss = train_loss_sum / train_samples
        train_accuracy = train_correct / train_samples

        # 검증에서는 평가 모드로 전환
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_samples = 0

        # 검증에서는 기울기를 추적하지 않음
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)

                val_loss_sum += loss.item() * batch_size

                predictions = logits.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_samples += batch_size

        val_loss = val_loss_sum / val_samples
        val_accuracy = val_correct / val_samples

        seconds = perf_counter() - start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "seconds": seconds,
        }
        history.append(record)

        # 완료한 epoch까지의 기록을 매번 CSV로 저장
        with (result_dir / "history.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(record.keys()),
            )
            writer.writeheader()
            writer.writerows(history)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss {train_loss:.4f} / {val_loss:.4f} | "
            f"accuracy {train_accuracy:.2%} / {val_accuracy:.2%} | "
            f"{seconds:.2f}초",
            flush=True,
        )

    # 학습·검증 곡선에 사용할 epoch 번호
    epoch_numbers = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 왼쪽 그래프: 손실
    axes[0].plot(
        epoch_numbers,
        [row["train_loss"] for row in history],
        label="train",
    )
    axes[0].plot(
        epoch_numbers,
        [row["val_loss"] for row in history],
        label="validation",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # 오른쪽 그래프: 정확도
    axes[1].plot(
        epoch_numbers,
        [row["train_accuracy"] for row in history],
        label="train",
    )
    axes[1].plot(
        epoch_numbers,
        [row["val_accuracy"] for row in history],
        label="validation",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    # 그래프 저장 후 메모리에서 그림 객체 정리
    fig.tight_layout()
    fig.savefig(result_dir / "learning_curves.png", dpi=150)
    plt.close(fig)

    # 마지막 epoch의 모델 상태와 복원에 필요한 설정 저장
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "TinyViT",
            "model_config": {
                "image_size": 32,
                "patch_size": 8,
                "embedding_dim": 64,
                "num_heads": 4,
                "depth": 2,
                "num_classes": 10,
            },
            "classes": dataset.classes,
            "input_size": [32, 32],
            "color_mode": "RGB",
            "preprocessing": "ToTensor",
            "epochs": epochs,
            "seed": seed,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
        },
        checkpoint_dir / "tiny_vit_cifar10.pt",
    )

    print(f"결과 저장: {result_dir}")
    print(f"모델 저장: {checkpoint_dir / 'tiny_vit_cifar10.pt'}")


if __name__ == "__main__":
    main()