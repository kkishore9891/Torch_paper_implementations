import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchinfo import summary
from vision_transformer_pytorch import VisionTransformer
from utils import train_loop, test_loop, checkpoint
 
training_data = datasets.FashionMNIST(
root="data",
train=True,
download=True,
transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

model = VisionTransformer(image_h=28, image_w=28, image_c=1, patch_d=2,
                            dropout=0.1, n_encoders=6, n_heads=4,
                            embedding_dim=64, ff_multiplier=2, n_classes=10,
                            device='cuda').to(device='cuda')

summary(model)

learning_rate = 3e-3
batch_size = 64
epochs = 10

trainloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                        shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                        shuffle=True)

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_test_accuracy = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(t+1, trainloader, batch_size, model, loss_fn, optimizer)
    test_accuracy = test_loop(t+1, testloader, model, loss_fn)
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        checkpoint(model=model, filename=f"models/fashion_mnist/best.pth")
print("Done!")