import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.models import vision_transformer
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from vision_transformer_pytorch import VisionTransformer
from utils import resume, infer_and_visualize
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

resume(model=model, filename="models/cat_dog/cat_dog_20250116-065955_best.pth")

class_names = ["Cat", "Dog"]

model = model.to(device=device)

# Print model summary
summary(model, input_size=(128, 3, 224, 224))


test_dataset = ImageFolder(root="data/Cats_vs_Dogs/test_set", transform=test_transform)

test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True, pin_memory=True)

infer_and_visualize(test_loader, model, class_names, device='cuda')

