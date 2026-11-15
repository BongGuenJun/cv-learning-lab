import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim=64, num_heads=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*4),
            nn.GELU(),
            nn.Linear(embedding_dim*4, embedding_dim),
        )

    def forward(self, tokens):
        # 각 토큰의 특징 축 정규화
        normalized = self.norm1(tokens)

        # 같은 입력에서 내부적으로 Q,K,V를 생성하는 Self_Attention
        attention_output, attention_weights = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights= False,
        )

        # 첫 번째 잔차 연결
        tokens = tokens + attention_output

        # 토큰별 특징 변환 후 두 번째 잔차 연결
        mlp_output = self.mlp(self.norm2(tokens))
        output = tokens + mlp_output

        return output, attention_weights


def main():
    torch.manual_seed(42)

    images = torch.randn(2,3,32,32)

    patch_embedding = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=8,
        stride=8,
    )

    tokens = patch_embedding(images).flatten(2).transpose(1,2)

    block = TransformerBlock(
        embedding_dim=64,
        num_heads=4,
    )

    block.eval()

    with torch.no_grad():
        output, attention_weights = block(tokens)

        print("입력 토큰:", tokens.shape)
        print("블록 출력:", output.shape)
        print("Head별 Attention 비중:", attention_weights.shape)


if __name__=="__main__":
    main()