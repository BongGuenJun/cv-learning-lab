def xyxy_to_yolo(box, image_width, image_height):
    x1, y1, x2, y2 = box

    if image_width <= 0 or image_height <= 0:
        raise ValueError("이미지 너비와 높이는 양수여야 합니다.")

    if not (
        0 <= x1 < x2 <= image_width
        and 0 <= y1 < y2 <= image_height
    ):
        raise ValueError("박스 좌표의 순서와 이미지 범위를 확인하세요.")

    # 두 모서리의 중간 지점
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    width = x2 - x1
    height = y2 - y1

    # 원본 이미지 크기를 기준으로 정규화
    return [
        center_x / image_width,
        center_y / image_height,
        width / image_width,
        height / image_height,
    ]


def yolo_to_xyxy(box, image_width, image_height):
    center_x, center_y, width, height = box

    # 정규화된 값을 픽셀 단위로 복원
    center_x *= image_width
    center_y *= image_height
    width *= image_width
    height *= image_height

    return [
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    ]


def main():
    image_width = 100
    image_height = 100

    # 예제 데이터셋에서 0을 고양이로 정의했다고 가정
    class_id = 0
    original_box = [10, 20, 50, 60]

    normalized_box = xyxy_to_yolo(
        original_box,
        image_width,
        image_height,
    )

    restored_box = yolo_to_xyxy(
        normalized_box,
        image_width,
        image_height,
    )

    # 객체 하나의 라벨 한 줄 생성
    label_line = (
        f"{class_id} "
        + " ".join(f"{value:.6f}" for value in normalized_box)
    )

    print("원본 xyxy:", original_box)
    print("YOLO 정규화 좌표:", normalized_box)
    print("라벨 한 줄:", label_line)
    print("복원한 xyxy:", restored_box)


if __name__ == "__main__":
    main()