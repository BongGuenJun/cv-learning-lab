import torch
from torch import nn

def main():
    torch.manual_seed(42)

    # RGB 32*32 이미지 2장 생성
    images = torch.randn(2,3,32,32)

    # 겹치지 않는 8*8 패치마다 64개 특징값 생성
    patch_embedding = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=8,
        stride=8,
    )

    features = patch_embedding(images)

    # 패치의 세로,가로 축을 하나로 합침
    flattened = features.flatten(start_dim=2)

    # [배치, 패치 수, 임베딩 차원] 순서로 변경
    tokens = flattened.transpose(1,2)

    print("입력 이미지:", images.shape)
    print("패치 임베딩:", features.shape)
    print("공간 축 펼치기:", flattened.shape)
    print("패치 토큰:", tokens.shape)
    print("첫 이미지의 첫 토큰:", tokens[0,0].shape)

if __name__=="__main__":
    main()