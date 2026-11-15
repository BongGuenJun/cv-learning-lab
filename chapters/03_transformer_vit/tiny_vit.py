import torch
from torch import nn

from transformer_block import TransformerBlock


class TinyViT(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=8,
        embedding_dim=64,
        num_heads=4,
        depth=2,
        num_classes=10,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                "이미지 크기는 패치 크기로 나누어떨어져야 합니다."
            )

        self.image_size = image_size
        num_patches = (image_size // patch_size) ** 2

        # RGB 패치마다 특징 벡터 생성
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 이미지마다 앞에 붙일 학습 가능한 CLS 토큰
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embedding_dim)
        )

        # CLS와 각 패치 위치에 대응하는 학습 가능한 벡터
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embedding_dim)
        )

        # 작은 난수로 초기화
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

        # 각 블록은 독립적인 가중치를 가짐
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, images):
        if images.shape[-2:] != (
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                f"입력 높이·너비는 {self.image_size}여야 합니다."
            )

        # [배치, 3, 32, 32] → [배치, 64, 4, 4]
        tokens = self.patch_embedding(images)

        # [배치, 64, 4, 4] → [배치, 16, 64]
        tokens = tokens.flatten(2).transpose(1, 2)

        batch_size = tokens.size(0)

        # 동일한 CLS 파라미터를 각 이미지에 사용
        # [1, 1, 64] → [배치, 1, 64]
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )

        # 토큰 축을 따라 CLS와 패치 토큰을 이어 붙임
        # [배치, 1, 64] + [배치, 16, 64]
        # → [배치, 17, 64]
        tokens = torch.cat(
            [cls_tokens, tokens],
            dim=1,
        )

        # 위치 벡터를 원소별로 더함
        # 배치마다 같은 위치 임베딩을 적용
        tokens = tokens + self.position_embedding

        # Attention과 MLP로 토큰 표현을 순차적으로 갱신
        for block in self.blocks:
            tokens, _ = block(tokens)

        tokens = self.norm(tokens)

        # 각 이미지의 첫 토큰인 CLS의 출력 특징 선택
        # [배치, 17, 64] → [배치, 64]
        cls_features = tokens[:, 0, :]

        # 이미지마다 클래스별 점수 10개 생성
        logits = self.head(cls_features)

        return logits


def main():
    torch.manual_seed(42)

    images = torch.randn(2, 3, 32, 32)

    model = TinyViT()
    model.eval()

    with torch.no_grad():
        logits = model(images)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("입력 이미지:", images.shape)
    print("CLS 파라미터:", model.cls_token.shape)
    print("위치 임베딩:", model.position_embedding.shape)
    print("Transformer 블록 수:", len(model.blocks))
    print("모델 출력:", logits.shape)
    print("파라미터 수:", parameter_count)


if __name__ == "__main__":
    main()