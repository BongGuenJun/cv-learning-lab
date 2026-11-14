import argparse
import csv
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights


def main():
    # 터미널에서 실험할 학습 범위 선택
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=["head", "layer4", "full", "layer4_dlr"],
        required=True,
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    seed = 42
    epochs = 10

    # 기존 3 epoch 파일을 덮어쓰지 않는 별도 결과 경로
    run_dir = (
        project_root
        / "results"
        / "finetuning"
        / f"{args.scope}_10epochs"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    weights = ResNet18_Weights.IMAGENET1K_V1

    dataset = datasets.CIFAR10(
        root=str(project_root / "data"),
        train=True,
        download=True,
        transform=weights.transforms(),
    )

    # 모든 실험에서 같은 데이터 선택
    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=split_generator,
    )

    train_dataset = Subset(dataset, indices[:200].tolist())
    val_dataset = Subset(dataset, indices[200:300].tolist())

    # 모든 실험에서 같은 배치 순서 사용
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    # 같은 사전학습 가중치와 같은 초기 분류기로 시작
    torch.manual_seed(seed)

    model = resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 10)

    # 선택한 범위의 동결 해제
    if args.scope in ["layer4", "layer4_dlr"]:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    elif args.scope == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    parameter_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    criterion = nn.CrossEntropyLoss()

    if args.scope == "layer4_dlr":
        # 파리미터 그룹마다 다른 학습을 적용
        optimizer = torch.optim.SGD(
            [
                {
                    "params":model.layer4.parameters(),
                    "lr":0.001,
                },
                {
                    "params":model.fc.parameters(),
                    "lr":0.01,
                },
            ],
            lr=0.01,
        )
    else:
        optimizer=torch.optim.SGD(
            trainable_parameters,
            lr=0.01
        )

    print("학습 범위:", args.scope)
    print("학습 가능 파라미터 수:", parameter_count)
    print("BatchNorm 이동 통계: 고정")

    history = []

    for epoch in range(1, epochs + 1):
        start = perf_counter()

        # 세 실험 모두 BatchNorm 이동 통계를 유지
        # 평가 모드 자체는 기울기 계산을 막지 않음
        model.eval()
        model.fc.train()

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
            "scope": args.scope,
            "trainable_parameters": parameter_count,
            "feature_learning_rate": optimizer.param_groups[0]["lr"],
            "head_learning_rate": optimizer.param_groups[-1]["lr"],
            "train_loss": train_loss_sum / train_samples,
            "val_loss": val_loss_sum / val_samples,
            "train_accuracy": train_correct / train_samples,
            "val_accuracy": val_correct / val_samples,
            "seconds": perf_counter() - start,
        }
        history.append(record)

        # 매 epoch 저장하여 중단되더라도 완료된 기록은 남김
        with (run_dir / "history.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(history[0].keys()),
            )
            writer.writeheader()
            writer.writerows(history)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss {record['train_loss']:.4f} / "
            f"{record['val_loss']:.4f} | "
            f"accuracy {record['train_accuracy']:.2%} / "
            f"{record['val_accuracy']:.2%} | "
            f"{record['seconds']:.2f}초",
            flush=True,
        )

    # 마지막 epoch의 모델 저장
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "resnet18",
            "classes": dataset.classes,
            "pretrained_weights": "IMAGENET1K_V1",
            "preprocessing": (
                "ResNet18_Weights.IMAGENET1K_V1.transforms"
            ),
            "training_scope": args.scope,
            "batchnorm_running_stats": "fixed",
            "epochs": epochs,
            "seed": seed,
            "feature_learning_rate" : optimizer.param_groups[0]["lr"],
            "hear_learning_rate" : optimizer.param_groups[-1]["lr"],
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
        },
        run_dir / "last.pt",
    )

    print("결과 저장:", run_dir)


if __name__ == "__main__":
    main()