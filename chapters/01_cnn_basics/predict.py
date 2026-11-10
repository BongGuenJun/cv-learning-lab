import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from model import SmallCNN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
checkpoint_path = PROJECT_ROOT / "checkpoints" / "smallcnn_cifar10.pt"

# 저장 파일을 cpu 메모리로 읽기
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

# 학습할 때와 같은 구조의 모델 생성
model = SmallCNN()

# 새 모델의 초기 가중치를 저장된 학습 가중치로 교체
model.load_state_dict(checkpoint["model_state_dict"])

# 추론을 위한 평가 모드 설정
model.eval()

print("모델 복원 완료")
print("클래스:", checkpoint["model_name"])
print("클래스:", checkpoint["classes"])
print("입력 크기:", checkpoint["input_size"])
print("색상 모드:", checkpoint["color_mode"])
print("전처리:", checkpoint["preprocessing"])
print("학습 epoch:", checkpoint["epochs"])

# 직접 준비한 이미지 경로
image_path = PROJECT_ROOT / "data" / "cat_test.jpg"

# 파일을 열고 저장된 입력 색상 모드로 변환
with Image.open(image_path) as source:
    image = source.convert(checkpoint["color_mode"])

# 저장된 입력 크기에 맞추고 학습 때와 같은 값 범위로 변환
transform = transforms.Compose([
    transforms.Resize(tuple(checkpoint["input_size"])),
    transforms.ToTensor(),
])

# [C,H,W] -> [N,C,H,W]
image_tensor = transform(image).unsqueeze(0)

# 복원한 모델로 이미지 한 장 추론
with torch.no_grad():
    logits = model(image_tensor)

# logits를 클래스 축으로 Softmax 변화
probabilities = torch.softmax(logits, dim=1)

class_index = probabilities.argmax(dim=1).item()
class_name = checkpoint["classes"][class_index]
score = probabilities[0, class_index].item()

print("logits:",logits)
print("이미지:", image_path)
print("입력 텐서 shape:", image_tensor.shape)
print("예측 클래스:", class_name)
print(f"Softmax 점수: {score:.2%}")