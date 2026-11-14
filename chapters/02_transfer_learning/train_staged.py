import argparse
import csv
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.models import ResNet18_Weights, resnet18


def main():
    # 단계적 동결 해제 실험 설정
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=["staged"],
        default="staged",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    seed = 42
    epochs = 10
    unfreeze_epoch = 6

    run_dir = (
        project_root
        / "results"
        / "finetuning"
        / f"{args.scope}_{epochs}epochs"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # 사전학습 가중치에 맞는 전처리 사용
    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    dataset = datasets.CIFAR10(
        root=str(project_root / "data"),
        train=True,
        download=True,
        transform=preprocess,
    )

    # 기존 실험과 동일한 이미지로 학습·검증 데이터 구성
    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=split_generator,
    )

    train_dataset = Subset(dataset, indices[:200])
    val_dataset = Subset(dataset, indices[200:300])

    # 학습 이미지가 섞이는 순서도 동일하게 설정
    loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    # 새 출력층의 초기 가중치를 동일하게 설정
    torch.manual_seed(seed)

    model = resnet18(weights=weights)

    # 처음에는 특징 추출부 전체를 고정
    for parameter in model.parameters():
        parameter.requires_grad = False

    # 새 출력층은 학습 가능한 상태로 생성됨
    model.fc = nn.Linear(
        model.fc.in_features,
        len(dataset.classes),
    )

    criterion = nn.CrossEntropyLoss()

    # 1~5번째 epoch에서는 출력층만 업데이트
    optimizer = torch.optim.SGD(
        model.fc.parameters(),
        lr=0.01,
    )

    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("학습 방식: 1~5 epoch는 fc, 6~10 epoch는 layer4 + fc")
    print(f"초기 학습 가능 파라미터 수: {trainable_count}")
    print("BatchNorm 이동 통계: 고정")

    history = []

    for epoch in range(1, epochs + 1):
        # 6번째 epoch부터 기존 layer4의 가중치도 학습
        if epoch == unfreeze_epoch:
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True

            # layer4와 fc를 모두 업데이트 대상으로 지정
            # fc의 기존 학습 가중치는 그대로 유지됨
            optimizer = torch.optim.SGD(
                [
                    {
                        "params": model.layer4.parameters(),
                        "lr": 0.01,
                    },
                    {
                        "params": model.fc.parameters(),
                        "lr": 0.01,
                    },
                ],
                lr=0.01,
            )

            trainable_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )

            print("\nlayer4 동결 해제: layer4 + fc 학습 시작")
            print(f"학습 가능 파라미터 수: {trainable_count}")

        start = perf_counter()

        # BatchNorm의 이동 평균·분산을 고정
        # eval 모드에서도 학습 가능한 파라미터는 업데이트 가능
        model.eval()
        model.fc.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

        train_loss = total_loss / total_samples
        train_accuracy = total_correct / total_samples

        # 검증에서는 기울기를 계산하거나 가중치를 업데이트하지 않음
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
                predictions = logits.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_samples += batch_size

        val_loss = val_loss_sum / val_samples
        val_accuracy = val_correct / val_samples

        seconds = perf_counter() - start

        # 특징 추출부가 고정된 구간에는 학습률을 빈칸으로 기록
        feature_learning_rate = (
            optimizer.param_groups[0]["lr"]
            if epoch >= unfreeze_epoch
            else None
        )
        head_learning_rate = optimizer.param_groups[-1]["lr"]

        record = {
            "epoch": epoch,
            "scope": args.scope,
            "trainable_parameters": trainable_count,
            "feature_learning_rate": feature_learning_rate,
            "head_learning_rate": head_learning_rate,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "seconds": seconds,
        }
        history.append(record)

        # 매 epoch마다 완료한 기록을 CSV에 저장
        with (run_dir / "history.csv").open(
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

    # 마지막 epoch의 모델과 실험 설정 저장
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "resnet18",
            "classes": dataset.classes,
            "pretrained_weights": "IMAGENET1K_V1",
            "preprocessing": (
                "ResNet18_Weights.IMAGENET1K_V1.transforms"
            ),
            "training_scope": "staged",
            "unfreeze_epoch": unfreeze_epoch,
            "batchnorm_running_stats": "fixed",
            "epochs": epochs,
            "seed": seed,
            "feature_learning_rate": feature_learning_rate,
            "head_learning_rate": head_learning_rate,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
        },
        run_dir / "last.pt",
    )

    print(f"결과 저장: {run_dir}")


if __name__ == "__main__":
    main()