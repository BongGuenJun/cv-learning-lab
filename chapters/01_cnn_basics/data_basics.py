from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallCNN

PROJECT_ROOT = Path(__file__).resolve().parents[2]

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.CIFAR10(
    root=str(PROJECT_ROOT / "data"),
    train=True,
    download=True,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

images, labels = next(iter(loader))

print("데이터 수:", len(dataset))
print("이미지 shape:", images.shape)
print("정답:", labels)
print("정답 이름:", [dataset.classes[label.item()] for label in labels])

model = SmallCNN()
logits = model(images)

print("모델 출력 shape:", logits.shape)