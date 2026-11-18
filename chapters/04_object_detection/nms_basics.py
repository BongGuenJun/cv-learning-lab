from bbox_basics import calculate_iou


def class_aware_nms(
    detections,
    confidence_threshold=0.25,
    iou_threshold=0.5,
):
    # 점수 기준을 통과한 후보만 새 리스트에 보관
    candidates = [
        detection
        for detection in detections
        if detection["score"] >= confidence_threshold
    ]

    # 점수가 높은 순서로 정렬
    candidates.sort(
        key=lambda detection: detection["score"],
        reverse=True,
    )

    selected = []

    while candidates:
        # 남은 후보 중 최고 점수 박스를 선택
        best = candidates.pop(0)
        selected.append(best)

        remaining = []

        for candidate in candidates:
            # 다른 클래스의 박스는 제거하지 않음
            if candidate["class_name"] != best["class_name"]:
                remaining.append(candidate)
                continue

            # 같은 클래스의 예측 박스끼리 겹침 비교
            iou = calculate_iou(
                best["box"],
                candidate["box"],
            )

            # 기준 이하로 겹치는 후보는 다음 비교를 위해 유지
            if iou <= iou_threshold:
                remaining.append(candidate)

        candidates = remaining

    return selected


def main():
    # 박스 형식은 모두 xyxy
    detections = [
        {
            "id": "A",
            "class_name": "cat",
            "score": 0.90,
            "box": [0, 0, 40, 40],
        },
        {
            "id": "B",
            "class_name": "cat",
            "score": 0.80,
            "box": [2, 2, 42, 42],
        },
        {
            "id": "C",
            "class_name": "cat",
            "score": 0.70,
            "box": [60, 60, 100, 100],
        },
        {
            "id": "D",
            "class_name": "dog",
            "score": 0.85,
            "box": [0, 0, 40, 40],
        },
        {
            "id": "E",
            "class_name": "cat",
            "score": 0.10,
            "box": [120, 120, 160, 160],
        },
    ]

    selected = class_aware_nms(
        detections,
        confidence_threshold=0.25,
        iou_threshold=0.5,
    )

    print("처리 전 후보 수:", len(detections))
    print(
        "처리 전 후보:",
        [detection["id"] for detection in detections],
    )

    print("\nNMS 이후 후보 수:", len(selected))

    for detection in selected:
        print(
            f'{detection["id"]} | '
            f'{detection["class_name"]} | '
            f'점수 {detection["score"]:.2f} | '
            f'박스 {detection["box"]}'
        )


if __name__ == "__main__":
    main()