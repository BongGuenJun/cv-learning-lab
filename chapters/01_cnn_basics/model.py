import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, channels=(8, 16)):
        super().__init__()

        # 두 합성곱 층의 출력 채널 수
        c1, c2 = channels

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(c2 * 8 * 8, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    images = torch.randn(4, 3, 32, 32)

    # 같은 입력을 두 구성에 전달하고 구조·파라미터 수 확인
    for channels in [(8, 16), (4, 8)]:
        model = SmallCNN(channels=channels)
        logits = model(images)

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        print("채널 구성:", channels)
        print("출력 shape:", logits.shape)
        print("파라미터 수:", parameter_count)
        print()