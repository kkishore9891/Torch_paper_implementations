import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from vision_transformer_pytorch import VisionTransformer
from utils import train_loop, test_loop, checkpoint
import wandb
import datetime

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations
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

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Initialize the model
model = VisionTransformer(
    image_h=224,
    image_w=224,
    image_c=3,
    patch_d=32,
    dropout=0.2,
    n_encoders=12,
    n_heads=12,
    embedding_dim=768,
    ff_multiplier=4,
    n_classes=2,
    device=device
).to(device)

# Print model summary
summary(model, input_size=(128, 3, 224, 224))

# Hyperparameters
learning_rate = 1e-4
batch_size = 128
epochs = 100

# Datasets and Dataloaders
train_dataset = ImageFolder(root="data/Cats_vs_Dogs/training_set", transform=train_transform)
test_dataset = ImageFolder(root="data/Cats_vs_Dogs/test_set", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialize the loss function
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize wandb
best_test_accuracy = 0
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
wandb.init(
    project="ViT_b_32",
    name=f"experiment_{curr_time}",
    config={
        "learning_rate": learning_rate,
        "architecture": "ViT_b_32",
        "dataset": "Cats_vs_Dogs",
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "loss_function": "NLLLoss"
    }
)

# Log model architecture
wandb.watch(model, log="all")

# Training Loop
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}/{epochs}\n{'-'*30}")
    
    # Training Phase
    train_loss, train_accuracy = train_loop(
        epoch=epoch,
        dataloader=train_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        wandb=wandb,
        device=device
    )
    
    # Testing Phase
    test_loss, test_accuracy = test_loop(
        epoch=epoch,
        dataloader=test_loader,
        model=model,
        loss_fn=loss_fn,
        wandb=wandb,
        device=device
    )
    
    # Checkpointing
    if test_accuracy > best_test_accuracy:
        print(f"New best accuracy: {test_accuracy:.2f}% (previous: {best_test_accuracy:.2f}%)")
        best_test_accuracy = test_accuracy
        checkpoint(model=model, filename="models/cat_dog/cat_dog_best.pth")
    
    print("\n")

print("Training Complete!")
wandb.finish()
