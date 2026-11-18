from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from evaluation_basics import evaluate_predictions


def main():
    project_root = Path(__file__).resolve().parents[2]
    result_dir = project_root / "results" / "detection_metrics"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 정답은 고양이 두 마리
    ground_truths = [
        {
            "class_name": "cat",
            "box": [0, 0, 40, 40],
        },
        {
            "class_name": "cat",
            "box": [60, 60, 100, 100],
        },
    ]

    predictions = [
        {
            "id": "A",
            "class_name": "cat",
            "score": 0.95,
            "box": [0, 0, 40, 40],
        },
        {
            "id": "B",
            "class_name": "cat",
            "score": 0.80,
            "box": [120, 120, 160, 160],
        },
        {
            "id": "C",
            "class_name": "cat",
            "score": 0.60,
            "box": [60, 60, 100, 100],
        },
    ]

    thresholds = [0.90, 0.70, 0.50]
    precisions = []
    recalls = []

    print("기준 | TP | FP | FN | Precision | Recall")

    for threshold in thresholds:
        selected = [
            prediction
            for prediction in predictions
            if prediction["score"] >= threshold
        ]

        result = evaluate_predictions(
            selected,
            ground_truths,
            iou_threshold=0.5,
        )

        precisions.append(result["precision"])
        recalls.append(result["recall"])

        print(
            f'{threshold:.2f} | '
            f'{result["tp"]} | '
            f'{result["fp"]} | '
            f'{result["fn"]} | '
            f'{result["precision"]:.2%} | '
            f'{result["recall"]:.2%}'
        )

    # 임계값별 Precision–Recall 지점 시각화
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        recalls,
        precisions,
        marker="o",
    )

    for threshold, recall, precision in zip(
        thresholds,
        recalls,
        precisions,
    ):
        ax.annotate(
            f"conf={threshold:.2f}",
            (recall, precision),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Operating Points")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True)

    fig.tight_layout()

    output_path = result_dir / "pr_operating_points.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("그래프 저장:", output_path)


if __name__ == "__main__":
    main()