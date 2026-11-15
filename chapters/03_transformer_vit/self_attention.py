import math

import torch
from torch import nn


def main():
    torch.manual_seed(42)

    # 이미지 2장을 각각 패치 토큰 16개로 변환
    images = torch.randn(2,3,32,32)

    embedding_dim = 64

    patch_embedding = nn.Conv2d(
        in_channels=3,
        out_channels=embedding_dim,
        kernel_size=8,
        stride=8,
    )

    features = patch_embedding(images)
    tokens = features.flatten(2).transpose(1,2)

    # 같은 토큰에서 서로 다른 Q, K, V 표현 생성
    query_layer = nn.Linear(embedding_dim, embedding_dim)
    key_layer = nn.Linear(embedding_dim, embedding_dim)
    value_layer = nn.Linear(embedding_dim, embedding_dim)

    queries = query_layer(tokens)
    keys = key_layer(tokens)
    values = value_layer(tokens)

    # 각 Query와 모든 Key 사이의 내적 계산
    # 마지막 두 축만 교환하여 이미지별로 행렬곱 수행
    scores = queries @ keys.transpose(-2,-1)

    # 특정 차원이 커질 때 점수 크기가 지나치게 커지는 것을 완화
    scores = scores / math.sqrt(embedding_dim)

    # 각 Query가 참고할 Key들의 비중 계산
    attention_weights = torch.softmax(scores, dim=-1)

    # 참고 비중으로 Value들을 가중함
    output = attention_weights @ values

    print("입력 토큰:", tokens.shape)
    print("Q:", queries.shape)
    print("K:", keys.shape)
    print("V:", values.shape)
    print("관계점수:", scores.shape)
    print("Attention 비중:", attention_weights.shape)
    print("Attention 출력:", output.shape)

    # 첫 이미지의 첫 토큰이 모든 토큰을 참고하는 비중
    first_token_weights = attention_weights[0,0]

    print(
        "첫 토큰의 참고 비중:",
        first_token_weights.detach(),
    )
    print(
        "첫 토큰의 참고 비중 합:",
        first_token_weights.sum().item(),
    )

if __name__ == "__main__":
    main()