import torch

# 흑백 이미지 4장, 높이 8, 너비 8을 가정한 연습 데이터
images = torch.arange(4*1*8*8, dtype=torch.float32)
images = images.reshape(4, 1, 8, 8)

print("전체 shape:", images.shape)
print("첫 이미지 shape:", images[0].shape)
print("첫 배치 shape:", images[:1].shape)
print("자료형:", images.dtype)
print("실행 장치:",images.device)


rgb_images = torch.zeros(2, 3, 16, 16)

print("RGB 전체:", rgb_images.shape)
print("RGB 첫 이미지:", rgb_images[0].shape)
print("RGB 첫 배치:", rgb_images[:1].shape)

print("기존 원소 수:", images.numel())
images.reshape(4, 3, 8, 8)