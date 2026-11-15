from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ViT_B_16_Weights, vit_b_16


def main():
    project_root = Path(__file__).resolve().parents[2]
    image_path = project_root / "data" / "cat_test.jpg"

    # 사용할 이미지가 있는지 확인
    if not image_path.is_file():
        raise FileNotFoundError(
            f"이미지를 찾을 수 없습니다: {image_path}"
        )

    # ImageNet 사전학습 가중치와 대응하는 전처리
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    # 모델 구조를 생성하고 사전학습 가중치 적용
    print("사전학습 ViT 불러오는 중...", flush=True)
    model = vit_b_16(weights=weights)
    model.eval()

    # 이미지 파일을 읽고 RGB로 변환
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0)

    # 기울기 추적 없이 이미지 한 장 추론
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

    # 점수가 높은 클래스 5개 선택
    top_scores, top_indices = probabilities[0].topk(5)
    class_names = weights.meta["categories"]

    print(f"\n이미지: {image_path}")
    print("입력 shape:", image_tensor.shape)
    print("출력 shape:", logits.shape)
    print("\n상위 5개 예측:")

    for score, index in zip(
        top_scores.tolist(),
        top_indices.tolist(),
    ):
        print(f"- {class_names[index]}: {score:.2%}")


if __name__ == "__main__":
    main()