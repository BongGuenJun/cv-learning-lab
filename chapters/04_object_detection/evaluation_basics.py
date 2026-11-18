from bbox_basics import calculate_iou


def evaluate_predictions(
    predictions,
    ground_truths,
    iou_threshold=0.5,
):
    # 높은 점수의 예측부터 정답과 매칭
    sorted_predictions = sorted(
        predictions,
        key=lambda prediction: prediction["score"],
        reverse=True,
    )

    matched_ground_truths = set()
    records = []

    true_positives = 0
    false_positives = 0

    for prediction in sorted_predictions:
        best_index = None
        best_iou = -1.0

        for index, ground_truth in enumerate(ground_truths):
            # 이미 다른 예측과 연결된 정답은 제외
            if index in matched_ground_truths:
                continue

            # 같은 클래스의 정답끼리 비교
            if prediction["class_name"] != ground_truth["class_name"]:
                continue

            iou = calculate_iou(
                prediction["box"],
                ground_truth["box"],
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            matched_ground_truths.add(best_index)
            true_positives += 1
            status = "TP"
        else:
            false_positives += 1
            status = "FP"

        records.append(
            {
                "id": prediction["id"],
                "score": prediction["score"],
                "status": status,
            }
        )

    false_negatives = (
        len(ground_truths) - len(matched_ground_truths)
    )

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    precision = (
        true_positives / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positives / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "records": records,
    }


def main():
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

    # Confidence 필터와 NMS 이후의 예측을 가정
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
            "box": [10, 10, 50, 50],
        },
        {
            "id": "C",
            "class_name": "dog",
            "score": 0.70,
            "box": [60, 60, 100, 100],
        },
    ]

    result = evaluate_predictions(
        predictions,
        ground_truths,
        iou_threshold=0.5,
    )

    for record in result["records"]:
        print(
            f'{record["id"]} | '
            f'점수 {record["score"]:.2f} | '
            f'{record["status"]}'
        )

    print(
        f'\nTP: {result["tp"]}, '
        f'FP: {result["fp"]}, '
        f'FN: {result["fn"]}'
    )
    print(f'Precision: {result["precision"]:.2%}')
    print(f'Recall: {result["recall"]:.2%}')


if __name__ == "__main__":
    main()