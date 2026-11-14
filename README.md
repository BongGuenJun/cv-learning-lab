# CV Learning Lab

컴퓨터 비전의 개념, 모델 구조, 학습·추론과 비교 실험을 기록하는 학습 저장소입니다.
챕터별·모델별·기술별로 정리하며, 추가 학습에 따라 계속 확장합니다.

## 개발 환경

- Ubuntu / WSL
- VS Code / WSL 연결
- Python / 프로젝트별 가상환경
- PyTorch / torchvision / matplotlib
- CPU 실행

## 학습 진행

| 챕터 | 내용 | 상태 |
|---|---|---|
| 1 | 텐서와 작은 CNN 기초 | 완료 |
| 2 | 학습·검증과 지표 시각화 | 완료 |
| 3 | 모델 저장·복원과 외부 이미지 추론 | 완료 |
| 4 | 채널 수에 따른 통제된 비교 실험 | 완료 |
| 5 | ResNet과 전이학습·파인튜닝 | 실험 완료 |
| 이후 | Transformer·ViT, 객체 탐지, 추론 활용과 Docker 등 | 예정 |

실험 결과는 기록된 데이터·학습 조건에서 관찰한 값이다.
모델의 일반적인 성능 순위나 최적 설정을 입증하는 결과로 해석하지 않는다.

<details>
<summary>챕터별 정리</summary>

<details>
<summary>챕터 1 — 텐서와 작은 CNN 기초</summary>

### 학습 내용

- 이미지 배치의 `[N, C, H, W]` 구조
- 인덱싱·슬라이싱의 차원 유지 차이
- 원소 수를 유지하는 reshape
- 합성곱·ReLU·MaxPooling
- Flatten·Linear를 통한 클래스별 점수 출력
- nn.Module·forward·모듈 import
- CIFAR-10 전처리·DataLoader

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
- [합성곱](chapters/01_cnn_basics/conv_basics.py)
- [모델](chapters/01_cnn_basics/model.py)
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
- backward: 기울기 계산
- step: 파라미터 업데이트

모델과 optimizer는 epoch 반복문 밖에서 생성하여 학습 상태를 유지한다.

### 학습과 검증

- train()은 학습 모드 설정이며, 학습 실행 자체가 아니다.
- 매 epoch 학습 후 검증셋을 평가한다.
- 검증에서는 eval()과 no_grad()를 사용한다.
- 검증에서는 역전파와 가중치 업데이트를 하지 않는다.

### 실습 설정

| 항목 | 설정 |
|---|---|
| 데이터 | CIFAR-10 |
| 학습 / 검증 | 공식 학습셋에서 겹치지 않게 선택한 2,000장 / 1,000장 |
| 입력 | RGB 32×32, ToTensor |
| 채널 | 8·16 |
| 배치 크기 | 32 |
| 학습 횟수 | 10 epoch |
| 손실 함수 | CrossEntropyLoss |
| Optimizer | SGD, lr=0.01 |
| 시드 | 42 |
| 장치 | CPU |

### 지표 집계

배치 평균 loss에 실제 배치 크기를 곱해 누적한 뒤,
전체 이미지 수로 나누어 평균 loss를 계산한다.
accuracy는 정답 수를 전체 이미지 수로 나눈다.

학습 지표는 업데이트 도중 배치별 결과를 누적한 값이고,
검증 지표는 해당 epoch 종료 후 모델로 계산한 값이다.

### 확인한 내용

학습·검증을 실행하여 loss 감소와 accuracy 상승을 확인했다.
지표를 CSV와 그래프로 저장했다.
이번 실습의 목적은 학습·검증 흐름과 시각화 구현이다.

### 코드와 결과

- [한 배치 업데이트](chapters/01_cnn_basics/train_step.py)
- [학습·검증·시각화](chapters/01_cnn_basics/train.py)
- [지표 CSV](results/01_cnn_basics/history.csv)

![학습 곡선](results/01_cnn_basics/learning_curves.png)

</details>

<details>
<summary>챕터 3 — 모델 저장·복원과 외부 이미지 추론</summary>

### 저장과 복원

state_dict와 클래스·입력 설정을 함께 저장한다.

```python
torch.save(checkpoint, checkpoint_path)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

model = SmallCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

- state_dict에는 가중치·편향과 버퍼 등 모델 상태가 들어간다.
- 모델 구조의 Python 코드 자체를 저장하는 것은 아니다.
- 복원할 때 호환되는 구조를 생성한 뒤 저장된 상태를 적용한다.
- 현재는 마지막 epoch 상태를 저장하며 optimizer 상태는 포함하지 않는다.
- 전처리 이름을 문자열로 기록해도 자동 실행되지는 않는다.

### 외부 이미지 적용

이미지 읽기 → RGB 변환 → 32×32 Resize → ToTensor
→ 배치 차원 추가 → 모델 추론 → 클래스·점수 출력

```python
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    logits = model(image_tensor)

probabilities = torch.softmax(logits, dim=1)
class_index = probabilities.argmax(dim=1).item()
class_name = checkpoint["classes"][class_index]
```

### 확인한 내용

별도 predict.py에서 모델을 복원하고 고양이 사진을 입력했다.

- 입력 shape: `[1, 3, 32, 32]`
- 예측 클래스: cat
- Softmax 점수: 13.02%

이번 실습의 목적은 저장한 모델을 외부 이미지에 적용하는 흐름의 구현이다.
Softmax 점수는 실제 정답 확률을 보장하지 않는다.

### 실행

프로젝트 루트에서 실행한다.
이미지 경로는 predict.py에서 지정한다.

```bash
source .venv/bin/activate

# 가중치가 없으면 먼저 학습·저장
python chapters/01_cnn_basics/train.py

# 저장된 가중치로 추론
python chapters/01_cnn_basics/predict.py
```

가중치와 개인 이미지는 Git에서 제외한다.

</details>

<details>
<summary>챕터 4 — 채널 수에 따른 통제된 비교 실험</summary>

### 질문

SmallCNN의 특징 채널을 8·16에서 4·8로 줄이면
파라미터 수·학습 결과·CPU 추론시간이 어떻게 달라지는가?

### 비교 조건

| 항목 | 설정 |
|---|---|
| 비교 구성 | 채널 8·16 / 채널 4·8 |
| 데이터 | 동일한 CIFAR-10 학습 2,000장·검증 1,000장 |
| 입력 | RGB 32×32, ToTensor |
| 배치 | 32, 두 모델의 학습 배치 순서 동일 |
| 학습 | 각각 새 모델로 100 epoch |
| Optimizer | SGD, lr=0.01 |
| 시드 | 42 |
| 조기 종료 | 사용하지 않음 |
| 저장 모델 | 100 epoch의 마지막 상태 |

채널 변경에 맞춰 다음 합성곱의 입력 채널과
Linear의 입력 특징 수도 변경했다.

### 100 epoch 결과

| 항목 | 기준 8·16 | 축소 4·8 |
|---|---:|---:|
| 파라미터 수 | 11,642 | 5,538 |
| 최종 학습 loss | 0.8822 | 1.2034 |
| 최종 검증 loss | 1.8076 | 1.8000 |
| 최종 학습 accuracy | 70.10% | 58.65% |
| 최종 검증 accuracy | 42.80% | 40.60% |
| 학습·검증 총시간 | 55.02초 | 40.90초 |

초기 10 epoch 실행에서는 검증 accuracy가 각각 26.7%, 15.6%였다.
100 epoch에서는 차이가 11.1%p에서 2.2%p로 줄었다.
짧은 학습 결과만으로 구조의 우열을 단정하기 어렵다는 점을 확인했다.
현재 결과 폴더에는 100 epoch 결과를 저장했다.

### CPU 추론시간 측정

| 조건 | 설정 |
|---|---|
| 입력 | 동일한 `[1, 3, 32, 32]` 텐서 |
| 장치·자료형 | CPU·float32 |
| CPU 연산 스레드 | 1개 |
| 실행 모드 | eval·no_grad |
| 사전 실행 | 모델마다 30회 |
| 측정 | 100회 호출을 1묶음으로 10묶음 |
| 순서 | 묶음마다 모델 측정 순서 교대 |
| 포함 범위 | model(images) 순전파 |
| 제외 범위 | 모델 로딩·파일 읽기·전처리·Softmax |

| 항목 | 기준 8·16 | 축소 4·8 |
|---|---:|---:|
| 호출당 평균 시간의 중앙값 | 0.1150 ms | 0.0807 ms |
| 묶음별 평균 시간 범위 | 0.1025~0.1226 ms | 0.0759~0.0894 ms |

중앙값은 개별 호출 시간이 아니라,
각 묶음의 호출당 평균 시간 10개에서 구한 값이다.

### 해석

이번 조건에서 축소 모델은 기준 모델보다:

- 파라미터 수가 약 52.4% 감소했다.
- 학습·검증 시간이 약 25.7% 감소했다.
- CPU 순전파 시간이 약 29.8% 감소했다.
- 최종 검증 accuracy는 2.2%p 낮았다.

채널 축소에 따른 모델 크기·계산 비용과 분류 결과의 절충을 확인했다.
파라미터 수 감소가 실행시간 감소에 그대로 비례하지는 않았다.

후반에는 학습 loss 감소와 달리 검증 loss가 최저점보다 높아지는
과적합 징후도 관찰했다. 최저 검증 loss 시점은 기준 모델 74 epoch,
축소 모델 90 epoch였으며 해당 가중치를 별도로 저장하지는 않았다.

### 한계와 후속 후보

- 단일 seed·분할의 학습용 실험이다.
- 동일한 seed가 다른 구조의 초기 가중치를 같게 만들지는 않는다.
- 결과를 다른 데이터·장치·학습 설정으로 일반화하지 않는다.
- 측정한 순전파 시간은 실제 서비스의 전체 응답시간이 아니다.
- RGB·흑백·개별 색상 채널 비교는 아직 수행하지 않았다.

### 코드와 결과

- [채널 설정이 가능한 모델](chapters/01_cnn_basics/model.py)
- [비교 학습](chapters/01_cnn_basics/compare_channels.py)
- [추론시간 측정](chapters/01_cnn_basics/benchmark_channels.py)
- [학습 결과 요약](results/channel_comparison/summary.csv)
- [기준 모델 학습 기록](results/channel_comparison/channels_8_16_history.csv)
- [축소 모델 학습 기록](results/channel_comparison/channels_4_8_history.csv)
- [추론시간 기록](results/channel_comparison/inference_benchmark.csv)

</details>

<details>
<summary>챕터 5 — ResNet과 전이학습·파인튜닝</summary>

### 학습 목표

- ResNet18의 특징 추출과 분류 흐름 이해
- 사전학습 가중치와 대응하는 전처리 사용
- 1,000클래스 출력층을 10클래스 출력층으로 교체
- 학습 범위·학습률·동결 해제 시점·BatchNorm 정책 비교
- 저장한 모델을 외부 이미지에 적용

### ResNet18의 기본 흐름

| 단계 | 출력 shape |
|---|---|
| 입력 | `[1, 3, 224, 224]` |
| conv1 | `[1, 64, 112, 112]` |
| bn1 → relu | `[1, 64, 112, 112]` |
| maxpool | `[1, 64, 56, 56]` |
| layer1 | `[1, 64, 56, 56]` |
| layer2 | `[1, 128, 28, 28]` |
| layer3 | `[1, 256, 14, 14]` |
| layer4 | `[1, 512, 7, 7]` |
| avgpool | `[1, 512, 1, 1]` |
| flatten | `[1, 512]` |
| fc | `[1, 1000]` |

avgpool은 채널별 공간 정보를 평균으로 집계한다.
flatten은 배치를 유지하면서 `[N, 512]`로 펼친다.
fc는 512개 특징을 받아 클래스별 logits를 출력한다.

### 전이학습 설정

```python
weights = ResNet18_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

model = resnet18(weights=weights)

for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 10)
```

기존 fc를 제거하고 새 fc로 교체한다.
기존 1,000클래스에서 10개를 골라내는 작업이 아니다.
새 fc는 초기화된 상태에서 CIFAR-10의 10개 클래스를 학습한다.

전처리는 짧은 변 256 Resize, 224×224 CenterCrop,
텐서 변환과 ImageNet 평균·표준편차 기반 정규화를 사용했다.

### 공통 실험 조건

| 항목 | 설정 |
|---|---|
| 데이터 | CIFAR-10 공식 학습셋에서 학습 200장·검증 100장 |
| 분할 | 서로 겹치지 않는 동일 인덱스 |
| 초기 가중치 | ImageNet 사전학습 ResNet18 |
| 출력층 | Linear(512, 10), 시드 42로 초기화 |
| 입력 | RGB 224×224 |
| 배치 크기 | 16 |
| 학습 횟수 | 각 10 epoch |
| 손실 함수 | CrossEntropyLoss |
| Optimizer | SGD, momentum 미사용 |
| 시드 | 42 |
| 장치 | CPU |
| 모델 저장 | 마지막 epoch 상태 |

각 실험은 동일한 사전학습 가중치와 새 출력층 초기화에서 시작했다.
앞 실험의 학습 결과를 이어받지 않았다.

### 실험별 변경 조건

| 실험 | 학습 범위 | 특징 추출부 학습률 | fc 학습률 | BatchNorm 이동 통계 |
|---|---|---:|---:|---|
| head | fc | 해당 없음 | 0.01 | 고정 |
| layer4 | layer4 + fc | 0.01 | 0.01 | 고정 |
| full | 전체 파라미터 | 0.01 | 0.01 | 고정 |
| layer4_dlr | layer4 + fc | 0.001 | 0.01 | 고정 |
| staged | 1~5회 fc, 6~10회 layer4 + fc | 개방 후 0.01 | 0.01 | 고정 |
| layer4_bn_update | layer4 + fc | 0.01 | 0.01 | 학습 중 전체 갱신 |

학습 범위를 넓혀도 모델 구조나 전체 파라미터 수는 바뀌지 않는다.
업데이트를 허용한 파라미터 수가 달라진다.

| 범위 | 학습 가능 파라미터 수 |
|---|---:|
| fc | 5,130 |
| layer4 + fc | 8,398,858 |
| 전체 | 11,181,642 |

### 10 epoch 결과

| 실험 | 최종 학습 accuracy | 최종 검증 accuracy | 최종 검증 loss |
|---|---:|---:|---:|
| head | 88.5% | 62% | 1.1355 |
| layer4 | 100% | 66% | 0.9546 |
| full | 49% | 14% | 3.0563 |
| layer4_dlr | 96% | 63% | 0.9480 |
| staged | 100% | 66% | 0.8317 |
| layer4_bn_update | 98% | 58% | 1.1709 |


</details>

</details>