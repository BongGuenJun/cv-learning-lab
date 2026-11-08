import torch
from torch import nn

images = torch.randn(4,3,32,32)

conv = nn.Conv2d(
    in_channels=3,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1,
)

features = conv(images)

relu = nn.ReLU()
pool = nn.MaxPool2d(kernel_size=2, stride=2)

activate = relu(features)
pooled = pool(activate)

print("입력 shape:", images.shape)
print("출력 shape:", features.shape)
print("필터 shape:", conv.weight.shape)
print("편향 shape:", conv.bias.shape)