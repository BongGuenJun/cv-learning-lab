from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import SmallCNN

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


torch.manual_seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

dataset = datasets.CIFAR10(
    root=str(PROJECT_ROOT / "data"),
             train=True,
             download=True,
             transform=transforms.ToTensor(),
             
)

# 고정된 무작위 순서에서 2,000장 선택
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(dataset), generator=generator)
train_dataset = Subset(dataset, indices[:2000].tolist())
val_dataset = Subset(dataset,indices[2000:3000].tolist())

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

val_loader =DataLoader(
    val_dataset,
    batch_size=32,
    shuffle = False,
    num_workers=0,
)

# 모델, 손실 함수, 가중치 업데이트 방법 설정
model = SmallCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 10
history = []

# 결과 저장 폴더 생성
# parents=True: 중간 폴더도 함께 생성
# exist_ok=True: 이미 폴더가 있어도 오류를 내지 않음
results_dir = PROJECT_ROOT / "results" / "01_cnn_basics"
results_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(1, epochs + 1):
    # 현재 epoch의 학습·검증 소요시간 측정 시작
    start = perf_counter()

    # ── 학습 ──
    # 직전 검증에서 평가 모드로 바뀌었으므로 다시 학습 모드로 설정
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_samples = 0

    for images, labels in train_loader:
        # 이전 배치의 기울기 초기화
        optimizer.zero_grad()

        # 순전파와 손실 계산
        logits = model(images)
        loss = criterion(logits, labels)

        # 역전파로 기울기 계산 후 가중치 업데이트
        loss.backward()
        optimizer.step()

        # 마지막 배치는 설정한 배치 크기보다 작을 수 있음
        batch_size = labels.size(0)

        # 배치 평균 loss를 배치의 loss 합으로 환산하여 누적
        train_loss_sum += loss.item() * batch_size

        # 이미지마다 가장 높은 점수를 가진 클래스 번호 선택
        predictions = logits.argmax(dim=1)

        # 정답 수와 처리한 이미지 수 누적
        train_correct += (predictions == labels).sum().item()
        train_samples += batch_size

    # ── 검증 ──
    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_samples = 0

    # 검증에서는 역전파가 필요 없으므로 기울기 추적 중지
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            val_loss_sum += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_samples += batch_size

    # 현재 epoch의 평균 지표와 소요시간 기록
    # accuracy는 0~1 사이의 비율로 저장
    record = {
        "epoch": epoch,
        "train_loss": train_loss_sum / train_samples,
        "val_loss": val_loss_sum / val_samples,
        "train_accuracy": train_correct / train_samples,
        "val_accuracy": val_correct / val_samples,
        "seconds": perf_counter() - start,
    }
    history.append(record)

    # 각 지표는 학습 / 검증 순서로 출력
    print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"loss {record['train_loss']:.4f} / {record['val_loss']:.4f} | "
        f"accuracy {record['train_accuracy']:.2%} / "
        f"{record['val_accuracy']:.2%} | "
        f"{record['seconds']:.2f}초"
    )

# ── 학습이 모두 끝난 뒤 지표 저장 ──
csv_path = results_dir / "history.csv"

# "w": 새로 작성하며, 기존 파일이 있으면 덮어씀
# newline="": CSV 작성 시 불필요한 줄바꿈 방지
# with 문이 끝나면 파일을 자동으로 닫음
with csv_path.open("w", newline="", encoding="utf-8") as file:
    # 딕셔너리의 키를 CSV 열 이름으로 사용
    writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
    writer.writeheader()
    writer.writerows(history)

# ── 학습 곡선 생성 ──
# history에서 epoch 번호만 추출해 가로축으로 사용
epoch_numbers = [row["epoch"] for row in history]

# 한 행에 그래프 두 개 생성
# fig: 전체 그림, axes: 개별 그래프 영역
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 왼쪽: 학습·검증 loss
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

# 오른쪽: 학습·검증 accuracy
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

# 정확도는 0~1 범위로 표시
axes[1].set_ylim(0, 1)
axes[1].legend()

# 그래프 제목과 축이 겹치지 않도록 배치 조정
fig.tight_layout()

# 그래프를 이미지로 저장하고 그림 객체 정리
figure_path = results_dir / "learning_curves.png"
fig.savefig(figure_path, dpi=150)
plt.close(fig)

print("CSV 저장:", csv_path)
print("그래프 저장:", figure_path)