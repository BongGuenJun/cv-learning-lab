from pathlib import Path

import ultralytics
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[2]

    model_path = project_root / "checkpoints" / "yolov8n.pt"
    result_root = project_root / "results" / "yolo_training"

    result_root.mkdir(parents=True, exist_ok=True)

    # 앞서 다운로드한 사전학습 모델에서 시작
    model = YOLO(str(model_path))

    print("Ultralytics 버전:", ultralytics.__version__)
    print("시작 가중치:", model_path)
    print("데이터: COCO8, 학습 4장·검증 4장")
    print("목적: 학습·검증·저장 흐름 확인")

    # 실제 학습 루프는 Ultralytics에서 실행
    model.train(
        data="coco8.yaml",
        epochs=10,
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,
        optimizer="SGD",
        lr0=0.001,
        seed=42,
        deterministic=True,
        amp=False,
        patience=0,
        project=str(result_root),
        name="yolov8n_coco8",
        exist_ok=False,
        plots=True,
        save=True,
    )

    # 실제 저장 경로 확인
    save_dir = Path(model.trainer.save_dir)

    print("\n학습 완료")
    print("결과 폴더:", save_dir)
    print("지표 CSV:", save_dir / "results.csv")
    print("최고 검증 평가 기준 가중치:", save_dir / "weights" / "best.pt")
    print("마지막 가중치:", save_dir / "weights" / "last.pt")


if __name__ == "__main__":
    main()