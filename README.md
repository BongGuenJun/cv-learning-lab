# CV Learning Lab

컴퓨터 비전의 개념, 모델 구현, 학습·추론과 비교 실험을 기록하는 학습 저장소입니다.
학습 내용을 챕터별·모델별·기술별로 찾아볼 수 있도록 정리합니다.

## 개발 환경

- Ubuntu / WSL
- VS Code / WSL 연결
- Python / 프로젝트별 가상환경
- PyTorch / torchvision / matplotlib
- CPU 실행

## 챕터별 정리

<details>
<summary>챕터별 학습 기록</summary>

<details>
<summary>챕터 1 — 텐서와 작은 CNN 기초 · 완료</summary>

### 학습 내용

- 이미지 텐서의 `[N, C, H, W]` 구조
- 인덱싱과 슬라이싱의 차원 유지 차이
- 전체 원소 수를 유지하는 reshape
- 합성곱, ReLU, MaxPooling
- Flatten과 Linear를 이용한 클래스별 점수 출력
- nn.Module, forward, 모듈 import와 직접 실행 구분
- CIFAR-10 전처리와 DataLoader

### 모델 흐름

| 단계 | 출력 shape |
|---|---|
| 이미지 배치 | `[4, 3, 32, 32]` |
| Conv → ReLU → Pool | `[4, 8, 16, 16]` |
| Conv → ReLU → Pool | `[4, 16, 8, 8]` |
| Flatten | `[4, 1024]` |
| Linear | `[4, 10]` |

### 코드

- [텐서 기초](chapters/01_cnn_basics/tensor_basics.py)
- [합성곱 기초](chapters/01_cnn_basics/conv_basics.py)
- [SmallCNN](chapters/01_cnn_basics/model.py)
- [데이터 로딩](chapters/01_cnn_basics/data_basics.py)

</details>

<details>
<summary>챕터 2 — 학습·검증과 지표 시각화 · 완료</summary>

### 학습 내용

- CrossEntropyLoss로 logits와 정답을 비교
- 역전파로 기울기를 계산하고 SGD로 가중치 업데이트
- 학습·검증 데이터 분리
- 학습 모드와 평가 모드, 기울기 추적 여부 구분
- epoch별 loss·accuracy 집계
- CSV 저장과 matplotlib 학습 곡선 생성

### 학습 순서

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

`model.train()`은 학습 모드 설정이며, 실제 가중치는
`optimizer.step()`에서 변경된다.

### 검증 순서

매 학습 epoch 종료 후 같은 검증셋을 평가한다.
`model.eval()`과 `torch.no_grad()`를 사용하며,
역전파와 가중치 업데이트는 수행하지 않는다.

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
| 실행 장치 | CPU |

### 확인한 결과

학습과 매 epoch 검증을 실행하여 loss 감소와 accuracy 상승을 확인했다.
지표를 CSV와 그래프로 저장했다.
이번 실습의 목적은 학습·검증 흐름과 시각화 구현이다.

학습 지표는 업데이트 도중 배치별 결과를 누적한 값이고,
검증 지표는 해당 epoch 종료 후 모델로 계산한 값이다.

### 코드와 결과

- [한 배치 가중치 업데이트](chapters/01_cnn_basics/train_step.py)
- [학습·검증·시각화](chapters/01_cnn_basics/train.py)
- [epoch별 지표 CSV](results/01_cnn_basics/history.csv)

![학습·검증 곡선](results/01_cnn_basics/learning_curves.png)

</details>

</details>

## 모델별 정리

<details>
<summary>모델별 학습 기록</summary>

<details>
<summary>SmallCNN — 직접 구성한 이미지 분류 CNN</summary>

### 구조

RGB 32×32 이미지를 입력받아 10개 클래스의 logits를 출력한다.

```text
Conv(3→8) → ReLU → MaxPool
→ Conv(8→16) → ReLU → MaxPool
→ Flatten → Linear(1024→10)
```

두 합성곱 모두 kernel=3, stride=1, padding=1이다.
MaxPooling은 kernel=2, stride=2이다.

### 특징과 제약

- 두 합성곱 블록으로 구성되어 층별 shape와 학습 흐름을 관찰하기 쉽다.
- Flatten 이후 Linear의 입력이 1,024개로 고정되어 있어
  입력 크기 변경 시 연결되는 특징 수를 확인해야 한다.
- 기본 블록을 구현하기 위한 학습용 모델이며,
  다른 모델과의 우열은 아직 비교하지 않았다.

### 현재 학습 범위

CIFAR-10 일부 데이터로 학습·검증하고 지표를 시각화했다.
가중치 저장·복원과 외부 이미지 추론은 다음 챕터에서 진행한다.

[모델 코드](chapters/01_cnn_basics/model.py)

</details>

</details>

## 기술별 정리

<details>
<summary>기술별 학습 기록</summary>

<details>
<summary>텐서 shape와 reshape</summary>

- 이미지 배치: `[N, C, H, W]`
- `images[0]`: 배치 차원 제거
- `images[:1]`: 배치 차원 유지
- reshape는 전체 원소 수를 유지하며 형태만 변경한다.
- shape의 각 축은 텐서의 용도에 따라 의미가 달라진다.

</details>

<details>
<summary>합성곱·ReLU·MaxPooling</summary>

### 합성곱

일반적인 groups=1 합성곱의 가중치 shape는
`[출력 채널, 입력 채널, 커널 높이, 커널 너비]`이다.

`Conv2d(3, 8, kernel_size=3)`의 가중치는 `[8, 3, 3, 3]`이다.
각 출력 필터는 입력 3채널의 계산 결과를 합하고 편향을 더해
출력 특징맵 하나를 만든다. 같은 가중치를 배치의 모든 이미지에 적용한다.

### ReLU

`max(0, x)`로 음수를 0으로 바꾸며 비선형성을 추가한다.
shape는 유지되고, 학습하는 파라미터는 없다.

### MaxPooling

채널별 지역 최댓값을 선택해 공간 크기를 축소한다.
이후 연산 부담을 줄이지만 세부 위치·특징 정보를 잃을 수 있다.
학습하는 파라미터는 없다.

</details>

<details>
<summary>Flatten·Linear·logits</summary>

- Flatten은 배치를 유지하고 이미지별 특징을 펼친다.
- Linear는 특징을 클래스별 점수로 변환한다.
- logits는 확률이 아니며 음수도 가능하다.
- `logits.argmax(dim=1)`은 클래스 축에서 최대 점수의 위치를 선택한다.
- CrossEntropyLoss에 전달할 때 별도 Softmax를 붙이지 않는다.

</details>

<details>
<summary>Dataset·전처리·DataLoader</summary>

- Dataset: 이미지와 정답을 제공한다.
- CIFAR-10의 `train=True`: 공식 학습용 분할을 선택한다.
- ToTensor: CIFAR-10 이미지를 `[C, H, W]` 텐서로 변환하고
  픽셀값을 0~255에서 0~1로 바꾼다.
- Subset: 원본 Dataset에서 지정한 인덱스만 사용한다.
- DataLoader: 데이터를 배치로 묶는다.
- `num_workers=0`: 별도 작업 프로세스 없이 현재 프로세스에서 읽는다.
- 마지막 배치는 설정된 배치 크기보다 작을 수 있다.

</details>

<details>
<summary>손실 함수·역전파·Optimizer</summary>

- 클래스 번호 형태의 정답은 이미지마다 하나이며 torch.long을 사용한다.
- CrossEntropyLoss는 기본 설정에서 배치 평균 loss를 반환한다.
- `zero_grad()`: 이전 기울기 초기화
- `backward()`: 기울기 계산
- `step()`: Optimizer 규칙에 따라 파라미터 수정
- `model.parameters()`: 모델에 등록된 파라미터들을 제공한다.
- 기본 SGD: `새 가중치 = 현재 가중치 - 학습률 × 기울기`
- `detach().clone()`: 기울기 추적에서 분리한 복사본을 만든다.

</details>

<details>
<summary>학습·검증 모드와 지표 집계</summary>

- `train()`은 학습 모드 설정이며 학습 실행 자체가 아니다.
- `eval()`은 평가 모드 설정이며 기울기 추적을 중지하지 않는다.
- `no_grad()`는 기울기 추적을 중지하며 평가 모드를 설정하지 않는다.
- 검증에서는 역전파와 가중치 업데이트를 하지 않는다.
- 전체 평균 loss는 `배치 평균 loss × 실제 배치 크기`를 누적한 뒤
  전체 이미지 수로 나누어 계산한다.
- accuracy는 `정답 수 / 전체 이미지 수`로 계산한다.
- loss는 점수 분포를 반영하고, accuracy는 최종 예측의 정오를 반영하므로
  매번 같은 방향으로 움직이지 않을 수 있다.

</details>

<details>
<summary>matplotlib과 CSV 기록</summary>

- epoch별 지표를 딕셔너리로 만들어 history 리스트에 누적한다.
- csv.DictWriter로 열 이름과 지표를 저장한다.
- `plt.subplots(1, 2)`로 그래프 두 개를 생성한다.
- `plot()`으로 곡선을 그리고 `legend()`로 범례를 표시한다.
- Agg 백엔드로 창 없이 이미지 파일을 저장한다.
- `savefig()`로 저장한 뒤 `plt.close(fig)`로 그림 객체를 정리한다.

</details>

</details>