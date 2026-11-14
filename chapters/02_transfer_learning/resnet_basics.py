import torch
from torchvision.models import resnet18

# 이번에는 구조만 관찰하므로 사전학습 가중치는 사용하지 않음
model = resnet18(weights=None)
model.eval()

images = torch.randn(1, 3, 224, 224)

# ReNet의 주요 단계를 순서대로 실행
with torch.no_grad():
    x = images
    print("입력:", x.shape)

    for name, layer in model.named_children():
        # torchvision ResNet 은 avgpool과 fc 사이에서 펼침을 수행
        if name == "fc":
            x = torch.flatten(x, start_dim=1)
            print("flatten:", x.shape)

        x = layer(x)
        print(f"{name}:", x.shape)

parameter_count = sum(
    parameter.numel()
    for  parameter in model.parameters()
)

print("\n파라미터 수:", parameter_count)
print("\n첫 잔차 블록:")
print(model.layer1[0])