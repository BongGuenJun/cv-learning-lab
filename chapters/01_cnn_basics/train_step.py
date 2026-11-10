import torch
from torch import nn

from model import SmallCNN

torch.manual_seed(42)

model = SmallCNN()
model.train()

# 학습 흐름을 확인하기 위한 가상 이미지와 정답
images = torch.randn(4,3,32,32)
labels = torch.tensor([5,2,6,3],dtype=torch.long)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 업데이터 전 가중치 일부를 별도로 복사
before = model.features[0].weight[0,0].detach().clone()

optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()

after = model.features[0].weight[0,0].detach().clone()

print("모델 출력 shape:", logits.shape)
print("loss:", loss.item())
print("업데이트 전 필터:\n", before)
print("업데이트 후 필터:\n", after)
print("가중치 변화량 합:",(after - before).abs().sum().item())