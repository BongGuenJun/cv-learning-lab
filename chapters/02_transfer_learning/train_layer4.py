from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42

torch.manual_seed(SEED)

# 사전학습 가중치와 그에 맞는 전처리
weights = ResNet18_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

dataset = datasets.CIFAR10(
    root=str(PROJECT_ROOT / "data"),
    train=True,
    download=True,
    transform=preprocess,
)

# train_head.py와 같은 학습·검증 데이터 선택
split_generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(len(dataset), generator=split_generator)

train_dataset = Subset(dataset, indices[:200].tolist())
val_dataset = Subset(dataset, indices[200:300].tolist())

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

# 같은 사전학습 가중치와 같은 초기 분류기에서 시작
torch.manual_seed(SEED)

model = resnet18(weights=weights)

# 먼저 기존 파라미터 전체 동결
for parameter in model.parameters():
    parameter.requires_grad = False

# 새 분류기는 기본적으로 학습 가능
model.fc = nn.Linear(model.fc.in_features, 10)

# 마지막 특징 추출 블록도 학습 허용
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("학습 범위: layer4 + fc")
print("학습 가능 파라미터 수:", trainable_parameters)

criterion = nn.CrossEntropyLoss()

# 학습이 허용된 파라미터만 optimizer에 전달
optimizer = torch.optim.SGD(
    [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ],
    lr=0.01,
)

epochs = 3

for epoch in range(1, epochs + 1):
    start = perf_counter()

    # BatchNorm의 이동 통계는 기존 분류기 학습과 동일하게 유지
    # 평가 모드여도 requires_grad=True인 파라미터는 학습 가능
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

    # 검증
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

    print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"loss {train_loss_sum / train_samples:.4f} / "
        f"{val_loss_sum / val_samples:.4f} | "
        f"accuracy {train_correct / train_samples:.2%} / "
        f"{val_correct / val_samples:.2%} | "
        f"{perf_counter() - start:.2f}초"
    )

# 마지막 모델 상태 저장
checkpoint_dir = PROJECT_ROOT / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = checkpoint_dir / "resnet18_layer4.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_name": "resnet18",
        "classes": dataset.classes,
        "pretrained_weights": "IMAGENET1K_V1",
        "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms",
        "training_scope": "layer4+fc",
        "batchnorm_running_stats": "fixed",
        "epochs": epochs,
        "seed": SEED,
    },
    checkpoint_path,
)

print("모델 저장:", checkpoint_path)