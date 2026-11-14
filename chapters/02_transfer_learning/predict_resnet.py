from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[2]
image_path = PROJECT_ROOT / "data" / "cat_test.jpg"

# 학습 때 사용한 전처리 준비
weights = ResNet18_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

# 이미지 파일을 열고 RGB로 변환
with Image.open(image_path) as source:
    image = source.convert("RGB")

# 전처리 후 배치 차원 추가: [3, 224, 224] → [1, 3, 224, 224]
images = preprocess(image).unsqueeze(0)

# 두 체크포인트를 차례대로 복원하고 같은 이미지에 적용
for checkpoint_name in ["resnet18_head.pt", "resnet18_layer4.pt"]:
    checkpoint_path = PROJECT_ROOT / "checkpoints" / checkpoint_name

    # 저장 파일 읽기
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    classes = checkpoint["classes"]

    # 저장 당시와 동일한 구조 생성
    # 전체 가중치를 체크포인트에서 복원하므로 사전학습 다운로드 불필요
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    # 학습된 가중치·편향·BatchNorm 상태 복원
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 추론: 가중치를 업데이트하지 않음
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)

    # 가장 높은 점수의 클래스 선택
    class_index = probabilities.argmax(dim=1).item()
    class_name = classes[class_index]
    score = probabilities[0, class_index].item()

    print("\n체크포인트:", checkpoint_name)
    print("학습 범위:", checkpoint["training_scope"])
    print("입력 shape:", images.shape)
    print("예측 클래스:", class_name)
    print(f"Softmax 점수: {score:.2%}")