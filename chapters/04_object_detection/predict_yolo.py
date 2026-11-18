import csv
from pathlib import Path

import ultralytics
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[2]

    image_path = project_root / "data" / "cat_test.jpg"
    checkpoint_dir = project_root / "checkpoints"
    result_dir = project_root / "results" / "yolo_inference"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"이미지를 찾을 수 없습니다: {image_path}"
        )

    # 공식 사전학습 가중치가 없으면 자동 다운로드
    model_path = checkpoint_dir / "yolov8n.pt"
    model = YOLO(str(model_path))

    # 전처리부터 예측 후처리까지 라이브러리에서 수행
    results = model.predict(
        source=str(image_path),
        device="cpu",
        imgsz=640,
        conf=0.05,
        iou=0.5,
        agnostic_nms=False,
        save=False,
        verbose=False,
    )

    # 이미지 한 장의 결과 선택
    result = results[0]
    boxes = result.boxes

    print("Ultralytics 버전:", ultralytics.__version__)
    print("모델:", model_path.name)
    print("이미지:", image_path)
    print("원본 이미지 크기 (높이, 너비):", result.orig_shape)
    print("탐지 객체 수:", len(boxes))

    records = []

    for index, box in enumerate(boxes, start=1):
        # 원본 이미지 기준 픽셀 좌표
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        confidence = float(box.conf.item())

        print(
            f"{index} | {class_name} | "
            f"점수 {confidence:.2%} | "
            f"xyxy [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )

        records.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    # 박스와 클래스가 그려진 이미지 저장
    output_image_path = result_dir / "prediction.jpg"
    result.save(filename=str(output_image_path))

    # 탐지되지 않아도 열 이름이 있는 CSV를 저장
    with (result_dir / "predictions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print("결과 이미지:", output_image_path)
    print("결과 CSV:", result_dir / "predictions.csv")


if __name__ == "__main__":
    main()