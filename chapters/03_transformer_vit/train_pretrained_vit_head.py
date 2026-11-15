import csv
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets
from torchvision.models import ViT_B_16_Weights, vit_b_16


def extract_features(model, loader, name):
    feature_batches = []
    label_batches = []

    model.eval()

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(
            loader,
            start=1,
        ):
            features = model(images)

            feature_batches.append(features)
            label_batches.append(labels)

            if batch_index % 10 == 0 or batch_index == len(loader):
                print(
                    f"{name} 특징 추출: "
                    f"{batch_index}/{len(loader)} 배치",
                    flush=True,
                )

    return TensorDataset(
        torch.cat(feature_batches, dim=0),
        torch.cat(label_batches, dim=0),
    )


def main():
    project_root = Path(__file__).resolve().parents[2]

    seed = 42
    epochs = 10

    result_dir = project_root / "results" / "vit_b16_head"
    checkpoint_dir = project_root / "checkpoints"

    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    dataset = datasets.CIFAR10(
        root=str(project_root / "data"),
        train=True,
        download=True,
        transform=preprocess,
    )

    # 이전 ResNet 전이학습 실험과 같은 인덱스 사용
    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=split_generator,
    )

    train_dataset = Subset(dataset, indices[:200])
    val_dataset = Subset(dataset, indices[200:300])

    # 특징 추출에서는 순서를 섞지 않음
    train_image_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    val_image_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    torch.manual_seed(seed)

    model = vit_b_16(weights=weights)

    # 기존 모델의 모든 파라미터 고정
    for parameter in model.parameters():
        parameter.requires_grad = False

    feature_dim = model.heads.head.in_features

    # 기존 1,000클래스 분류기를 제거하고 특징을 그대로 반환
    model.heads.head = nn.Identity()

    # CIFAR-10 분류를 위한 새 출력층
    head = nn.Linear(feature_dim, len(dataset.classes))

    print(f"이미지당 특징 수: {feature_dim}")
    print(
        "학습 가능 파라미터 수:",
        sum(parameter.numel() for parameter in head.parameters()),
    )

    # 고정된 특징을 한 번 추출하여 메모리에 보관
    extraction_start = perf_counter()

    train_features = extract_features(
        model,
        train_image_loader,
        "학습",
    )
    val_features = extract_features(
        model,
        val_image_loader,
        "검증",
    )

    extraction_seconds = perf_counter() - extraction_start

    print(f"특징 추출 총시간: {extraction_seconds:.2f}초")

    loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_features,
        batch_size=16,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_features,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)

    history = []

    for epoch in range(1, epochs + 1):
        start = perf_counter()

        head.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        # 이 반복문에서는 이미지 대신 추출한 특징을 받음
        for features, labels in train_loader:
            optimizer.zero_grad()

            logits = head(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            train_loss_sum += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_samples += batch_size

        train_loss = train_loss_sum / train_samples
        train_accuracy = train_correct / train_samples

        head.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for features, labels in val_loader:
                logits = head(features)
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

    # 학습한 분류기를 원래 모델의 출력 위치에 연결
    model.heads.head = head
    model.eval()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "vit_b_16",
            "classes": dataset.classes,
            "pretrained_weights": "IMAGENET1K_V1",
            "preprocessing": (
                "ViT_B_16_Weights.IMAGENET1K_V1.transforms"
            ),
            "training_scope": "head",
            "feature_cache": True,
            "feature_extraction_seconds": extraction_seconds,
            "feature_extraction_batch_size": 4,
            "head_batch_size": 16,
            "epochs": epochs,
            "seed": seed,
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
        },
        checkpoint_dir / "vit_b16_head.pt",
    )

    print(f"결과 저장: {result_dir}")
    print(f"모델 저장: {checkpoint_dir / 'vit_b16_head.pt'}")


if __name__ == "__main__":
    main()