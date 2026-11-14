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

# 학습.검증 인덱스를 중복 없이 선택
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

# 기존 특징 추출부의 파라미터 동결
model = resnet18(weights=weights)

for parameter in model.parameters():
    parameter.requires_grad = False

# 새 분류기 생성
model.fc = nn.Linear(model.fc.in_features, 10)

criterion = nn.CrossEntropyLoss()

# 업데이트 대상도 분류기의 파라미터로 한정
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)

epochs = 3

for epoch in range(1, epochs + 1):
    start = perf_counter()

    # 특징 추출부의 BatchNorm 이동 통계도 유지
    model.eval()

    # 새 분류기만 학습 모드로 설정
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

    # 검증에서는 전체 모델을 평가 모드로 설정
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

# 이후 복원할 수 있도록 마지막 모델 상태 저장
checkpoint_dir = PROJECT_ROOT / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = checkpoint_dir / "resnet18_head.pt"

torch.save(
    {
        "model_state_dict":model.state_dict(),
        "model_name": "resnet18",
        "classes" : dataset.classes,
        "pretrained_weights": "IMAGENET1K_V1",
        "preporcessing": "ResNet18_Weights.IMAGENET1K_V1.trainsforms",
        "training_scope": "fc",
        "epochs" : epochs,
        "seed" : SEED,
    },
    checkpoint_path,
)

print("모델 저장:", checkpoint_path)