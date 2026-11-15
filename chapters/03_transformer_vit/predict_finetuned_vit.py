from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def main():
    project_root = Path(__file__).resolve().parents[2]

    checkpoint_path = (
        project_root / "checkpoints" / "vit_b16_head.pt"
    )
    image_path = project_root / "data" / "cat_test.jpg"

    # 저장한 모델 상태와 클래스 정보 불러오기
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    classes = checkpoint["classes"]

    # 사전학습 가중치를 다시 다운로드하지 않고 구조만 생성
    model = vit_b_16(weights=None)

    # 저장된 가중치와 호환되도록 출력층을 10클래스로 교체
    model.heads.head = nn.Linear(
        model.heads.head.in_features,
        len(classes),
    )

    # 특징 추출부와 학습한 분류기의 상태를 모두 복원
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 학습 때 사용한 전처리 적용
    weights = ViT_B_16_Weights[
        checkpoint["pretrained_weights"]
    ]
    preprocess = weights.transforms()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0)

    # 새 이미지의 특징 추출과 분류를 함께 수행
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

    top_scores, top_indices = probabilities[0].topk(3)

    print(f"체크포인트: {checkpoint_path.name}")
    print(f"이미지: {image_path}")
    print("학습 범위:", checkpoint["training_scope"])
    print("입력 shape:", image_tensor.shape)
    print("출력 shape:", logits.shape)
    print("\n상위 3개 예측:")

    for score, index in zip(
        top_scores.tolist(),
        top_indices.tolist(),
    ):
        print(f"- {classes[index]}: {score:.2%}")


if __name__ == "__main__":
    main()