import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8,16,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(16*8*8, 10),
        )

    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x

if __name__=="__main__":
    model = SmallCNN()
    images = torch.randn(4,3,32,32)

    logits = model(images)

    print(model)
    print("입력:",images.shape)
    print("출력:",logits.shape)