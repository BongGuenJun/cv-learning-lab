from pathlib import Path

import torch
from torch import nn
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 어떤 사전학습 가중치를 사용하는지 명시
weights = ResNet18_Weights.IMAGENET1K_V1

# 해당 가중치에 맞는 전처리 가져오기
preprocess = weights.transforms()

# 첫 실행에서는 사전학습 가중치 다운로드
model = resnet18(weights=weights)

print("기존 분류기:", model.fc)
print("전처리:\n", preprocess)

# 사전학습된 모델의 기존 파라미터는 업데이트하지 않도록 설정
for parameter in model.parameters():
    parameter.requires_grad = False

# 마지막 분류기를 새로운 10클래스 분류기로 교체
# 새로 만든 층의 파라미터는 기본적으로 requires_grad=True
feature_count = model.fc.in_features
model.fc = nn.Linear(feature_count, 10)

print("교체한 분류기:", model.fc)

# CIFAR-10 이미지 한 장을 사전학습 전처리로 변환
dataset = datasets.CIFAR10(
    root=str(PROJECT_ROOT / "data"),
    train=True,
    download=True,
    transform=preprocess,
)

image, label = dataset[0]
images = image.unsqueeze(0)

model.eval()

with torch.no_grad():
    logits = model(images)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("입력 shape:", images.shape)
print("출력 shape:", logits.shape)
print("전체 파라미터 수:", total_parameters)
print("학습 가능 파라미터 수:", trainable_parameters)