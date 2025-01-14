import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from vision_transformer_pytorch import VisionTransformer
from utils import train_loop, test_loop, checkpoint

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
# the validation transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

model = VisionTransformer(image_h=224, image_w=224, image_c=3, patch_d=32,
                            dropout=0.2, n_encoders=12, n_heads=12,
                            embedding_dim=768, ff_multiplier=4, n_classes=2,
                            device='cuda').to(device='cuda')

summary(model)

learning_rate = 1e-5
batch_size = 2
epochs = 100

train_dataset = ImageFolder(root="data/Cats_vs_Dogs/training_set", transform=train_transform)
test_dataset = ImageFolder(root="data/Cats_vs_Dogs/test_set", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the loss function
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_test_accuracy = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(t+1,train_loader, batch_size, model, loss_fn, optimizer)
    test_accuracy = test_loop(t+1, test_loader, model, loss_fn)
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        checkpoint(model=model, filename=f"cat_dog_best.pth")
    checkpoint(model=model, filename=f"cat_dog_epoch-{t+1}.pth")
print("Done!")