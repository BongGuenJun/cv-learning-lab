def xywh_to_xyxy(box):
    # 왼쪽 위 좌표와 너비,높이를 두 모서리 좌표로 변환
    x,y,width,height = box

    return [
        x,
        y,
        x + width,
        y + height,
    ]

def calculate_iou(box_a, box_b):
    # 입력 박스 형식은 모두 xyxy
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # 교집합 사각형의 두 모서리
    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    # 겹치지 않는 경우 길이를 0으로 처리
    intersection_width = max(
        0,
        intersection_x2 - intersection_x1,
    )
    intersection_height = max(
        0,
        intersection_y2 - intersection_y1,
    )
    intersection_area = (
        intersection_width * intersection_height
    )

    # 각 박스의 면적
    area_a = (
        max(0, ax2 - ax1) * max(0, ay2 - ay1)
    )
    area_b = (
        max(0, bx2 - bx1) * max(0, by2 -by1)
    )

    union_area = area_a + area_b - intersection_area

    # 면적이 없는 박스끼리 비교하는 경우 처리
    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def main():
    original_box = [10, 20, 40, 40]
    converted_box = xywh_to_xyxy(original_box)

    print("왼쪽 위 기준 xywh:", original_box)
    print("변환한 xyxy:", converted_box)

    box_a = [0, 0, 40, 40]
    box_b = [20, 20, 60, 60]

    print("\n박스 A:", box_a)
    print("박스 B:", box_b)
    print(f"부분 겹침 IoU:, {calculate_iou(box_a, box_b):.4f}")
    print(f"동일 박스 IoU: {calculate_iou(box_a, box_a):.4f}")

    box_c = [50, 50, 70, 70]
    print(f"겹치지 않는 박스 IoU: {calculate_iou(box_a, box_c):.4f}")

if __name__=="__main__":
    main()