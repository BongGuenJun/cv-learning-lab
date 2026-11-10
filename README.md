# CV Learning Lab

컴퓨터 비전의 개념, 모델 구현, 학습·추론과 비교 실험을 기록하는 학습 저장소입니다.

<details>
<summary>챕터별 정리</summary>

<details>
<summary>챕터 1 — 텐서와 작은 CNN 기초</summary>

### 핵심 내용

- 이미지 배치의 기본 형태는 `[N, C, H, W]`이다.
- 인덱싱 `[0]`은 해당 차원을 제거하고, 슬라이싱 `[:1]`은 유지한다.
- reshape는 전체 원소 수를 유지하며 형태를 변경한다.
- 합성곱으로 특징을 계산하고 ReLU로 비선형성을 추가한다.
- MaxPooling으로 공간 크기를 축소한다.
- Flatten과 Linear로 이미지별 클래스 점수를 출력한다.
- Dataset·전처리·DataLoader를 통해 실제 이미지 배치를 준비한다.

### 모델 흐름

| 단계 | 출력 shape |
|---|---|
| 입력 | `[4, 3, 32, 32]` |
| Conv → ReLU → Pool | `[4, 8, 16, 16]` |
| Conv → ReLU → Pool | `[4, 16, 8, 8]` |
| Flatten | `[4, 1024]` |
| Linear | `[4, 10]` |

### 코드

- [텐서](chapters/01_cnn_basics/tensor_basics.py)
- [합성곱·활성화·Pooling](chapters/01_cnn_basics/conv_basics.py)
- [모델 정의](chapters/01_cnn_basics/model.py)
- [데이터 로딩](chapters/01_cnn_basics/data_basics.py)

</details>

<details>
<summary>챕터 2 — 학습·검증과 지표 시각화</summary>

### 학습 흐름

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

- zero_grad: 이전 기울기 초기화
- forward: 현재 가중치로 출력 계산
- loss: 출력과 정답 비교
- backward: 파라미터의 기울기 계산
- step: 기울기를 사용해 파라미터 업데이트

모델과 optimizer는 epoch 반복문 밖에서 생성하여 학습 상태를 유지한다.

### 학습과 검증

- 학습에서는 model.train()을 사용하고 가중치를 업데이트한다.
- 검증에서는 model.eval()과 torch.no_grad()를 사용한다.
- 매 학습 epoch 종료 후 같은 검증셋 전체를 평가한다.
- 검증에서는 backward와 optimizer.step을 실행하지 않는다.

### 실습 설정

| 항목 | 설정 |
|---|---|
| 데이터 | CIFAR-10 |
| 학습 / 검증 | 공식 학습셋에서 겹치지 않게 선택한 2,000장 / 1,000장 |
| 입력 | RGB, 32×32, ToTensor |
| 모델 | SmallCNN |
| 배치 크기 | 32 |
| 학습 횟수 | 10 epoch |
| 손실 함수 | CrossEntropyLoss |
| Optimizer | SGD, lr=0.01 |
| 난수 시드 | 42 |
| 장치 | CPU |

### 지표 계산

배치 평균 loss에 실제 배치 크기를 곱해 누적한 뒤,
전체 이미지 수로 나누어 평균 loss를 계산한다.

accuracy는 정답 수를 전체 이미지 수로 나누어 계산한다.

학습 지표는 업데이트 도중 배치별 결과를 누적한 값이고,
검증 지표는 해당 epoch 종료 후 모델로 계산한 값이다.

### 확인한 결과

학습과 매 epoch 검증을 실행하여 loss 감소와 accuracy 상승을 확인했다.
epoch별 지표를 CSV와 그래프로 저장했다.
이번 실습의 목적은 학습·검증 흐름과 시각화 구현이다.

### 코드와 결과

- [한 배치 업데이트](chapters/01_cnn_basics/train_step.py)
- [학습·검증·시각화](chapters/01_cnn_basics/train.py)
- [지표 CSV](results/01_cnn_basics/history.csv)

![학습·검증 곡선](results/01_cnn_basics/learning_curves.png)

</details>

<details>
<summary>챕터 3 — 모델 저장·복원과 외부 이미지 추론</summary>

### 모델 저장

model.state_dict()로 모델의 가중치·편향 등 상태를 가져오고,
torch.save()로 클래스 순서와 입력 설정을 함께 저장한다.

| 저장 항목 | 역할 |
|---|---|
| model_state_dict | 모델 가중치·편향 |
| model_name | 모델 이름 기록 |
| classes | 출력 번호에 대응하는 클래스 이름 |
| input_size | 입력 높이·너비 |
| color_mode | 입력 색상 모드 |
| preprocessing | 전처리 방법 기록 |
| epochs | 학습 횟수 기록 |

현재는 마지막 epoch의 상태를 저장한다.
최적 epoch 선택과 optimizer 상태 저장은 포함하지 않았다.

### 모델 복원

```python
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

model = SmallCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

- torch.load는 저장 파일을 읽는다.
- SmallCNN()은 가중치를 담을 모델 구조를 생성한다.
- load_state_dict는 초기 가중치를 저장된 값으로 교체한다.
- 모델 생성과 가중치 복원은 재학습이 아니다.
- state_dict에 모델 구조의 Python 코드 자체가 포함되는 것은 아니다.

### 외부 이미지 적용

```text
이미지 파일
→ RGB 변환
→ 32×32 크기 조정
→ ToTensor
→ 배치 차원 추가
→ 모델 추론
→ 클래스·Softmax 점수 출력
```

```python
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    logits = model(image_tensor)

probabilities = torch.softmax(logits, dim=1)
class_index = probabilities.argmax(dim=1).item()
class_name = checkpoint["classes"][class_index]
```

이미지 한 장도 배치 형태인 `[1, 3, 32, 32]`로 전달한다.

### 확인한 결과

학습 코드와 별도의 predict.py에서 저장된 모델을 복원했다.
직접 준비한 고양이 사진을 입력하여 다음 출력을 확인했다.

- 입력 텐서: `[1, 3, 32, 32]`
- 예측 클래스: cat
- Softmax 점수: 13.02%

이번 실습의 목적은 저장한 모델을 외부 이미지에 적용하는 흐름을 구현하는 것이다.

### 주의점

- 복원할 모델 구조와 저장된 가중치의 이름·shape가 호환되어야 한다.
- 저장한 전처리 문자열은 설명 정보이며 자동 실행되지 않는다.
- 추론에서도 학습에 맞는 색상·크기·값 범위를 준비해야 한다.
- 정사각형 Resize는 원본 이미지 비율을 바꿀 수 있다.
- Softmax 점수는 실제 정답 확률을 보장하지 않는다.
- 학습하지 않은 종류의 입력도 기존 10개 클래스 중 하나로 분류될 수 있다.

### 코드와 실행

- [학습·모델 저장](chapters/01_cnn_basics/train.py)
- [모델 복원·추론](chapters/01_cnn_basics/predict.py)

프로젝트 루트에서 실행한다.

```bash
source .venv/bin/activate

# 가중치가 없을 때 먼저 실행
python chapters/01_cnn_basics/train.py

# 이미지 경로를 predict.py에서 지정한 후 실행
python chapters/01_cnn_basics/predict.py
```

가중치와 개인 입력 이미지는 Git에서 제외한다.

</details>

</details>

<details>
<summary>모델별 정리</summary>

<details>
<summary>SmallCNN — 직접 구성한 이미지 분류 CNN</summary>

### 구조

```text
Conv(3→8) → ReLU → MaxPool
→ Conv(8→16) → ReLU → MaxPool
→ Flatten → Linear(1024→10)
```

- 입력: RGB 32×32 이미지 배치
- 출력: 이미지별 10개 클래스의 logits
- 합성곱: kernel=3, stride=1, padding=1
- MaxPooling: kernel=2, stride=2

### 특징과 제약

- 작은 구조로 층별 shape와 학습 흐름을 관찰하기 쉽다.
- Flatten 이후 Linear가 1,024개 특징을 받도록 고정되어 있다.
- 입력 크기나 채널을 바꾸면 연결되는 층의 shape도 확인해야 한다.
- 구조를 배우기 위한 모델이며 다른 모델과의 우열은 아직 비교하지 않았다.

### 학습과 적용

CIFAR-10 일부 데이터로 10 epoch 학습·검증하고 지표를 시각화했다.
가중치와 클래스·입력 설정을 저장한 뒤,
별도 프로그램에서 복원하여 외부 이미지의 클래스와 점수를 출력했다.

### 클래스 순서

```text
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

- [모델](chapters/01_cnn_basics/model.py)
- [학습](chapters/01_cnn_basics/train.py)
- [추론](chapters/01_cnn_basics/predict.py)

</details>

</details>

<details>
<summary>기술별 정리</summary>

<details>
<summary>텐서·배치·shape</summary>

- 이미지 배치는 `[N, C, H, W]` 순서로 사용한다.
- N은 이미지 수, C는 채널 수, H와 W는 높이·너비이다.
- images[0]은 배치 차원을 제거하고 images[:1]은 유지한다.
- reshape는 전체 원소 수를 유지하며 형태를 바꾼다.
- unsqueeze(0)은 맨 앞에 크기 1인 차원을 추가한다.
- 같은 숫자의 shape라도 텐서 용도에 따라 축의 의미가 다르다.

</details>

<details>
<summary>합성곱·가중치 구조</summary>

일반적인 groups=1 합성곱의 가중치는
`[출력 채널, 입력 채널, 커널 높이, 커널 너비]` 형태이다.

Conv2d(3, 8, kernel_size=3)의 가중치는 `[8, 3, 3, 3]`이다.
출력 필터마다 입력 채널별 계산을 합하고 편향을 더해 특징맵 하나를 만든다.
같은 필터를 배치의 모든 이미지에 적용하므로 가중치에는 배치 차원이 없다.

- weight[0]: 첫 출력 필터 전체
- weight[0, 0]: 첫 출력 필터의 첫 입력 채널용 커널
- features[0]: Sequential의 첫 번째 층

stride는 이동 간격, padding은 입력 가장자리의 확장을 설정한다.

</details>

<details>
<summary>ReLU·MaxPooling</summary>

### ReLU

max(0, x)로 비선형성을 추가한다.
shape는 유지되고 학습 파라미터는 없다.
음수 구간에서는 기울기가 0이다.

### MaxPooling

채널별 지역 최댓값을 선택하여 공간 크기를 줄인다.
이후 연산 부담을 줄이지만 작은 특징과 위치 정보가 사라질 수 있다.
채널을 서로 합치는 연산은 아니다.

</details>

<details>
<summary>Flatten·Linear·logits·Softmax</summary>

- Flatten은 배치를 유지하면서 이미지별 특징을 펼친다.
- Linear는 특징으로 클래스별 점수를 계산한다.
- logits는 확률이 아니며 음수도 가능하다.
- argmax(dim=1)은 클래스 축에서 최대 점수의 위치를 찾는다.
- Softmax는 클래스별 값을 합이 1인 분포로 변환한다.
- Softmax 점수가 실제 정답 확률을 보장하지는 않는다.
- CrossEntropyLoss 앞에 별도 Softmax를 붙이지 않는다.

</details>

<details>
<summary>Dataset·전처리·DataLoader</summary>

- Dataset은 이미지와 정답을 제공한다.
- CIFAR-10의 train=True는 공식 학습용 분할을 선택한다.
- Subset은 지정한 인덱스의 데이터만 사용한다.
- randperm은 정수 인덱스를 중복 없이 섞는다.
- ToTensor는 CIFAR-10 이미지를 `[C, H, W]` 텐서로 변환하고
  픽셀값을 0~255에서 0~1로 바꾼다.
- DataLoader는 데이터를 배치로 묶어 제공한다.
- num_workers=0은 현재 프로세스에서 데이터를 읽는다는 뜻이다.
- 마지막 배치는 설정한 배치 크기보다 작을 수 있다.
- transform은 전처리를 담은 변수명이며 다른 이름으로 바꿀 수 있다.
- torchvision.transforms는 이미지 변환 도구 모듈이다.
- Transformer는 Attention 기반 신경망 구조로, 전처리 transform과 다르다.

</details>

<details>
<summary>손실 함수·역전파·Optimizer</summary>

- 클래스 번호 형태의 정답은 이미지마다 하나이며 torch.long을 사용한다.
- CrossEntropyLoss는 현재 기본 설정에서 배치 평균 loss를 반환한다.
- zero_grad는 이전 기울기를 초기화한다.
- backward는 기울기를 계산하여 파라미터의 grad에 누적한다.
- step은 기울기를 사용해 파라미터를 수정한다.
- model.parameters()는 모델에 등록된 파라미터를 제공한다.
- 기본 SGD: `새 가중치 = 현재 가중치 - 학습률 × 기울기`
- detach().clone()은 기울기 추적에서 분리한 별도 복사본을 만든다.

</details>

<details>
<summary>학습·검증과 지표 집계</summary>

- train()은 학습 모드 설정이며 학습 실행 자체가 아니다.
- eval()은 평가 모드 설정이며 기울기 추적을 중지하지 않는다.
- no_grad()는 기울기 추적을 중지하며 평가 모드를 설정하지 않는다.
- 검증은 반복되지만 검증 데이터로 가중치를 업데이트하지 않는다.
- 전체 평균 loss는 배치 평균에 실제 배치 크기를 곱해 누적한 뒤
  전체 이미지 수로 나누어 계산한다.
- accuracy는 정답 수를 전체 이미지 수로 나눈다.
- loss와 accuracy는 반영하는 정보가 달라 매번 같이 움직이지 않을 수 있다.

</details>

<details>
<summary>CSV·matplotlib 시각화</summary>

- epoch별 지표를 딕셔너리로 만들어 리스트에 누적한다.
- csv.DictWriter로 열 이름과 지표를 저장한다.
- plt.subplots(1, 2)로 그래프 두 개를 생성한다.
- plot으로 곡선을 그리고 legend로 범례를 표시한다.
- Agg 백엔드로 창 없이 그림을 저장한다.
- savefig로 저장한 뒤 plt.close로 그림 객체를 정리한다.

</details>

<details>
<summary>체크포인트·모델 저장과 복원</summary>

- 체크포인트는 특정 시점의 모델 상태를 저장한 것이다.
- checkpoints라는 폴더명 자체에는 특별한 기능이 없다.
- state_dict는 파라미터와 버퍼 등의 상태를 이름과 텐서로 제공한다.
- torch.save로 저장하고 torch.load로 읽는다.
- load_state_dict로 생성된 모델에 저장 상태를 적용한다.
- .pt는 바이너리 파일이므로 일반 텍스트처럼 읽지 않는다.
- 확장자가 같아도 내부 저장 구조는 다를 수 있다.
- 현재 파일에는 optimizer 상태가 없어 전체 학습 재개 상태를 저장한 것은 아니다.

</details>

<details>
<summary>Python 모듈·직접 실행</summary>

- model.py는 모듈이고 SmallCNN은 그 안의 클래스이다.
- from model import SmallCNN으로 클래스 정의를 가져온다.
- Python은 일반적인 최상위 함수·클래스를 가져올 때 export가 필요 없다.
- SmallCNN()은 모델 인스턴스를 생성하며 학습된 가중치를 자동으로 읽지 않는다.
- 직접 실행한 파일의 __name__은 "__main__"이 된다.
- if __name__ == "__main__": 아래 코드는 직접 실행할 때만 수행된다.

</details>

</details>