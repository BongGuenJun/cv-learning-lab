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
| 6 | Transformer·ViT 구현, 학습과 외부 이미지 추론 | 실습 완료·복습 중 |
| 이후 | 객체 탐지, 추론 활용과 Docker 등 | 예정 |

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


<details>
<summary>챕터 6 — Transformer와 ViT</summary>

### 학습 범위

- 이미지의 패치 토큰 변환과 위치 임베딩
- Q·K·V와 단일 Head Self-Attention 계산
- Multi-Head Attention·LayerNorm·MLP·잔차 연결
- CLS 토큰을 사용하는 TinyViT 구현
- CIFAR-10 일부 데이터로 처음부터 학습·검증
- 사전학습 ViT 추론과 분류기 전이학습
- 체크포인트 복원과 직접 준비한 이미지 추론

현재까지 수행한 실습을 기록했다.
LoRA·Adapter 등 효율적 파인튜닝 개요는 미진행이며 후속 학습 항목이다.

### 파일별 역할

| 파일 | 역할 |
|---|---|
| [patch_embedding.py](chapters/03_transformer_vit/patch_embedding.py) | 이미지를 패치 토큰으로 변환 |
| [self_attention.py](chapters/03_transformer_vit/self_attention.py) | Q·K·V와 Attention 행렬 계산 |
| [transformer_block.py](chapters/03_transformer_vit/transformer_block.py) | Multi-Head Attention·MLP·정규화·잔차 연결 |
| [tiny_vit.py](chapters/03_transformer_vit/tiny_vit.py) | 작은 ViT 전체 구조 |
| [train_tiny_vit.py](chapters/03_transformer_vit/train_tiny_vit.py) | TinyViT 학습·검증·곡선·체크포인트 저장 |
| [predict_pretrained_vit.py](chapters/03_transformer_vit/predict_pretrained_vit.py) | ImageNet 사전학습 ViT로 외부 이미지 추론 |
| [train_pretrained_vit_head.py](chapters/03_transformer_vit/train_pretrained_vit_head.py) | 고정된 ViT 특징 추출과 새 분류기 학습 |
| [predict_finetuned_vit.py](chapters/03_transformer_vit/predict_finetuned_vit.py) | 저장 모델 복원과 CIFAR-10 클래스 추론 |

### 패치 임베딩

RGB 32×32 이미지를 겹치지 않는 8×8 패치로 처리했다.
패치는 이미지당 16개이며, 각 패치를 특징값 64개로 표현했다.

```python
patch_embedding = nn.Conv2d(3, 64, kernel_size=8, stride=8)
features = patch_embedding(images)
tokens = features.flatten(2).transpose(1, 2)
```

| 단계 | shape |
|---|---|
| 이미지 | `[2, 3, 32, 32]` |
| 패치 임베딩 | `[2, 64, 4, 4]` |
| 공간 축 펼치기 | `[2, 64, 16]` |
| 축 교환 | `[2, 16, 64]` |

최종 축의 의미는 이미지 2장, 패치 토큰 16개, 토큰당 특징 64개이다.
64는 클래스 수가 아니다.
축 교환은 이후 Attention의 입력 형식에 맞추는 과정이다.

### Q·K·V와 Self-Attention

같은 입력 토큰에 서로 다른 학습 가능한 선형 변환을 적용하여 Q·K·V를 만든다.

- Q: 현재 출력을 만드는 토큰 쪽의 비교 표현
- K: 참고 대상 토큰 쪽의 비교 표현
- V: 계산한 비중으로 실제 결합할 정보

```python
queries = query_layer(tokens)
keys = key_layer(tokens)
values = value_layer(tokens)

scores = queries @ keys.transpose(-2, -1)
scores = scores / math.sqrt(embedding_dim)
attention_weights = torch.softmax(scores, dim=-1)
output = attention_weights @ values
```

| 값 | shape | 의미 |
|---|---|---|
| Q·K·V 각각 | `[2, 16, 64]` | 토큰별 변환 결과 |
| 관계 점수·Attention 비중 | `[2, 16, 16]` | 각 토큰이 모든 토큰을 참고하는 관계 |
| 출력 | `[2, 16, 64]` | 다른 패치 정보가 결합된 토큰 |

Attention 비중은 마지막 축에서 합이 1이다.
분류 정답 확률이 아니라 V를 섞는 비율이다.
정답 라벨은 Attention에 직접 들어가지 않고, 최종 logits와 loss를 계산할 때 사용한다.

### Transformer 블록의 흐름

```mermaid
flowchart TD
    X["입력 토큰 X"] --> N1["LayerNorm 1"]
    N1 --> Q["Q 변환"]
    N1 --> K["K 변환"]
    N1 --> V["V 변환"]
    Q --> S["Q와 K 비교·크기 조정·Softmax"]
    K --> S
    S --> A["V 가중합·Head 결합·출력 변환"]
    V --> A
    X --> ADD1["첫 잔차 연결"]
    A --> ADD1
    ADD1 --> T["갱신된 토큰 T"]
    T --> N2["LayerNorm 2"]
    N2 --> M["MLP"]
    T --> ADD2["두 번째 잔차 연결"]
    M --> ADD2
    ADD2 --> O["출력 토큰"]
```

```python
normalized = self.norm1(tokens)

attention_output, attention_weights = self.attention(
    query=normalized,
    key=normalized,
    value=normalized,
    need_weights=True,
    average_attn_weights=False,
)

tokens = tokens + attention_output
output = tokens + self.mlp(self.norm2(tokens))
```

- 같은 텐서를 query·key·value 입력에 전달해도 내부 변환 가중치는 서로 다르다.
- `Linear(64, 64)`는 각 토큰의 마지막 특징 축을 변환한다. 토큰 수는 유지한다.
- 같은 구조의 Linear를 별도로 생성하면 각자 다른 파라미터를 가진다.
- 순전파 호출 자체로 가중치가 바뀌지는 않는다.
- seed 고정은 초기화 과정을 재현하는 것이며 모든 층을 같은 값으로 만드는 것이 아니다.
- Multi-Head Attention은 여러 변환 공간에서 관계를 계산한 뒤 결과를 결합한다.
- 임베딩 64·Head 4 설정에서 Head당 차원은 16이다.
- Head별 비중 shape는 `[배치, Head, Query 토큰, Key 토큰]`이다.
- 특정 Head가 색상이나 모양을 담당하도록 미리 지정하지 않는다.
- Attention은 토큰 사이의 정보를 결합한다.
- MLP는 같은 가중치를 각 토큰에 독립적으로 적용하여 특징을 가공한다.
- MLP는 `Linear(64,256) → GELU → Linear(256,64)` 순서로 특징을 변환한다.
- LayerNorm은 여기서 각 토큰의 마지막 특징 축을 정규화한다. BatchNorm 이동 통계는 없다.
- 잔차 연결은 같은 위치의 원소끼리 더하며 shape를 유지한다.

### Python 호출과 텐서 연산 복습

`block = TransformerBlock()`에서 초기화하고,
`block(tokens)`에서 준비된 모듈의 forward를 실행한다.
호출할 때마다 가중치를 다시 초기화하지 않는다.

함수 인자는 위치 또는 이름으로 전달할 수 있다.
Attention의 앞 세 입력은 query·key·value 순서이다.

| 연산 | 결과 예시 |
|---|---|
| Python 리스트의 `+` | `[1,2] + [3,4] → [1,2,3,4]` |
| 텐서의 `+` | `tensor([1,2]) + tensor([3,4]) → tensor([4,6])` |
| `torch.cat(..., dim=1)` | 토큰 축을 따라 이어 붙임 |
| `transpose(1,2)` | 두 축의 순서를 교환 |
| `expand(batch_size,-1,-1)` | 나머지 축을 유지하며 배치에 사용할 뷰로 확장 |

### TinyViT 전체 구조

```mermaid
flowchart TD
    I["이미지 · N×3×32×32"] --> P["패치 임베딩 · N×16×64"]
    P --> C["CLS 추가 · N×17×64"]
    C --> POS["위치 임베딩 더하기"]
    POS --> B["Transformer 블록 2개"]
    B --> N["LayerNorm"]
    N --> S["CLS 출력 선택 · N×64"]
    S --> H["Linear · 64에서 10"]
    H --> O["클래스별 logits · N×10"]
```

- CLS는 정답이 아니라, 분류에 사용할 정보를 학습하는 토큰이다.
- 위치 임베딩은 각 토큰 위치에 대응하는 학습 가능한 벡터이다.
- CLS와 위치 임베딩을 nn.Parameter로 등록하여 함께 학습한다.
- nn.ModuleList로 두 블록을 등록하며, 각 블록은 독립된 가중치를 가진다.
- CLS는 이어 붙이고 위치 임베딩은 원소별로 더한다.
- 마지막에는 `tokens[:, 0, :]`로 CLS 위치를 선택한다.
- 직접 구성한 모델의 파라미터 수는 114,250개이다.

### 정답이 학습에 연결되는 위치

```mermaid
flowchart TD
    I["이미지"] --> M["ViT"]
    M --> P["클래스별 logits"]
    P --> L["CrossEntropyLoss"]
    Y["정답 labels"] --> L
    L --> B["backward"]
    B --> U["optimizer.step"]
    U --> M
```

```python
logits = model(images)
loss = criterion(logits, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

역전파는 분류기뿐 아니라 패치 임베딩, Q·K·V 변환, MLP,
정규화의 학습 파라미터, CLS와 위치 임베딩까지 영향을 줄 수 있다.
Attention 비중은 입력에서 계산되는 값이며 독립적으로 저장된 학습 파라미터가 아니다.

### TinyViT 학습 결과

| 항목 | 설정·결과 |
|---|---|
| 초기화 | 사전학습 없이 시작 |
| 학습·검증 이미지 | 2,000장 / 1,000장 |
| 입력 | 32×32, ToTensor |
| 배치·epoch | 32 / 10 |
| Optimizer | AdamW, lr=0.001, weight_decay=0.01 |
| 시드 | 42 |
| 최종 학습 loss / accuracy | 1.5992 / 40.90% |
| 최종 검증 loss / accuracy | 1.9317 / 28.30% |

loss 감소와 accuracy 상승을 확인했다.
이번 목적은 모델 구조와 학습·검증 루프를 연결하는 것이다.
이전 SmallCNN과는 옵티마이저와 구조 등이 달라 구조만의 효과를 분리한 비교가 아니다.

[학습 기록](results/tiny_vit/history.csv)

![TinyViT 학습 곡선](results/tiny_vit/learning_curves.png)

### 사전학습 ViT와 분류기 학습

ViT-B/16의 B는 Base, 16은 패치 한 변의 크기를 뜻한다.
이번 가중치는 입력 224×224를 사용하며, 패치 196개와 CLS를 합쳐 토큰 197개를 처리한다.
CLS 출력 특징은 768차원이다.

| 항목 | 설정 |
|---|---|
| 모델·가중치 | torchvision ViT-B/16, IMAGENET1K_V1 |
| 데이터 | ResNet 전이학습과 동일 인덱스의 학습 200장·검증 100장 |
| 입력 | 가중치에 대응하는 전처리, RGB 224×224 |
| 학습 범위 | 새 Linear(768, 10)만 학습 |
| 학습 가능 파라미터 | 768×10 + 10 = 7,690개 |
| 학습 | 10 epoch, 배치 16, SGD lr=0.01 |
| 특징 추출 | 평가 모드·배치 4·기울기 추적 없음 |
| 시드 | 42 |

```mermaid
flowchart TD
    I["이미지"] --> V["고정된 사전학습 ViT"]
    V --> F["768차원 특징을 한 번 추출·보관"]
    F --> H["새 분류기"]
    H --> L["정답과 loss 계산"]
    L --> U["분류기만 업데이트"]
    U --> H
```

기존 분류기를 Identity로 바꿔 특징을 추출하고,
메모리에 보관한 특징으로 새 분류기를 반복 학습했다.
종료 후 학습한 분류기를 모델에 다시 연결하여 전체 state_dict를 저장했다.

고정된 특징 추출부, 평가 모드, 결정적인 전처리를 사용했기에 특징 재사용이 가능했다.
특징 추출부를 학습하거나 무작위 증강을 매번 적용하는 경우에는 같은 방식으로 대체할 수 없다.

| 항목 | 결과 |
|---|---:|
| 학습·검증 특징 추출 총시간 | 59.87초 |
| 분류기 학습·검증 시간 | epoch당 약 0.01~0.02초 |
| 최종 학습 loss / accuracy | 0.2471 / 100% |
| 최종 검증 loss / accuracy | 0.5748 / 91% |

epoch 시간에는 최초 특징 추출 시간이 포함되지 않는다.
이 값은 새 이미지의 추론시간이 아니며, ResNet의 전체 순전파 포함 epoch 시간과 직접 비교하지 않는다.
200장·100장의 단일 분할 결과로 모델의 일반적인 우열을 판단하지 않는다.
ResNet과 비교할 때 모델 크기·사전학습 가중치·특징 추출 방식 등의 차이도 명시해야 한다.

[사전학습 ViT 분류기 학습 기록](results/vit_b16_head/history.csv)

### 저장 모델과 외부 이미지 추론

```python
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

model = vit_b_16(weights=None)
model.heads.head = nn.Linear(
    model.heads.head.in_features,
    len(checkpoint["classes"]),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

weights=None은 구조 생성 시 사전학습 가중치를 가져오지 않는다는 뜻이다.
이후 체크포인트에 저장된 전체 모델 상태를 복원했다.
전처리는 학습 때 사용한 가중치 설정과 맞췄다.

| 외부 고양이 이미지 실험 | 출력 클래스 | 최고 점수 |
|---|---|---|
| ImageNet 사전학습 ViT 그대로 사용 | 1,000클래스 | Egyptian cat 64.95% |
| CIFAR-10으로 분류기 학습 후 복원 | 10클래스 | cat 63.08% |

두 결과는 클래스 구성이 다르므로 Softmax 점수를 직접 비교하지 않는다.
실제 품종을 확인하지 않았으므로 첫 결과의 품종 정답 여부는 판단하지 않는다.
Softmax 점수는 실제 정답 확률을 보장하지 않는다.

새 이미지는 보관해둔 학습 특징을 사용할 수 없으므로,
`model(image_tensor)`로 특징 추출과 분류를 모두 수행한다.

### CNN과 기본 ViT 비교

| 비교 축 | 일반적인 CNN | 기본 ViT |
|---|---|---|
| 표현 | 공간 특징맵 | 패치 토큰 |
| 주요 정보 결합 | 합성곱 | Self-Attention |
| 한 층의 연결 범위 | 커널 주변 | 전체 토큰 사이 |
| 결합 계수 | 학습한 커널을 위치마다 공유 | 입력에서 Q·K로 비중 계산 |
| 위치 관계 | 합성곱의 공간 배치에 반영 | 위치 임베딩 등으로 반영 |
| 주요 비용 요인 | 해상도·채널·커널 등 | 토큰 수·특징 차원·블록 수 등 |

전체 Attention의 관계 행렬은 토큰 수의 제곱 크기이다.
패치를 작게 하거나 입력 해상도를 높이면 토큰 수와 비용이 증가한다.
Swin처럼 지역 Attention을 사용하는 변형도 있으므로 모든 Transformer에 같은 비용 특성을 적용하지 않는다.
Conv2d를 패치 임베딩에 사용한다고 모델 전체가 CNN 계열이 되는 것은 아니다.

### 주요 모델 계열 지도

직접 모델 구현·학습을 실습한 범위와 특징만 개괄한 범위를 구분한다.
아래 목록은 전체 모델을 망라하거나 성능 순위를 정한 것이 아니다.

| 모델 | 계열·용도 | 핵심 특징 | 현재 학습 범위 |
|---|---|---|---|
| ResNet | CNN·분류/백본 | 잔차 연결로 깊은 구조 학습 지원 | 구조·전이학습 실습 |
| EfficientNet | CNN·분류/백본 | 깊이·너비·해상도를 함께 조절 | 특징 개요 |
| MobileNet | CNN·분류/백본 | 경량 연산과 배포 효율을 고려한 구조 | 특징 개요 |
| ConvNeXt | CNN·분류/백본 | Transformer 설계의 일부 아이디어를 반영한 현대적 CNN | 특징 개요 |
| ViT | Transformer·분류/백본 | 패치 토큰과 전체 Self-Attention | 작은 구조 구현·학습·사전학습 활용 |
| DeiT | ViT 기반 | 학습 전략과 증류로 데이터 효율 개선 | 특징 개요 |
| Swin Transformer | 계층형 Transformer·백본 | 윈도우 Attention·윈도우 이동·계층 구조 | 특징 개요 |
| DETR | Transformer 기반 탐지 | 원형은 CNN 백본과 Transformer encoder-decoder 결합 | 특징 개요·탐지 학습 예정 |

분류 백본과 객체 탐지 시스템은 역할이 다르다.
데이터 장수만으로 적합한 모델을 정하지 않고,
클래스별 다양성·사전학습과의 차이·목표 지표·학습 자원·배포 조건을 함께 고려한다.

### 실행

프로젝트 루트에서 실행한다.

```bash
python chapters/03_transformer_vit/patch_embedding.py
python chapters/03_transformer_vit/self_attention.py
python chapters/03_transformer_vit/transformer_block.py
python chapters/03_transformer_vit/tiny_vit.py
python chapters/03_transformer_vit/train_tiny_vit.py
python chapters/03_transformer_vit/predict_pretrained_vit.py
python chapters/03_transformer_vit/train_pretrained_vit_head.py
python chapters/03_transformer_vit/predict_finetuned_vit.py
```

가중치와 개인 이미지는 Git에서 제외한다.

### 참고 자료

- [ViT 논문](https://arxiv.org/abs/2010.11929)
- [DeiT 논문](https://proceedings.mlr.press/v139/touvron21a.html)
- [Swin Transformer 논문](https://arxiv.org/abs/2103.14030)
- [DETR 논문](https://arxiv.org/abs/2005.12872)
- [ConvNeXt 논문](https://arxiv.org/abs/2201.03545)
- [torchvision ViT 구현](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py)

</details>

</details>
