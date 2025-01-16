"""
This program contains useful functions required while creating the transformer module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m
from tqdm import tqdm


class PatchPositionEmbedding(nn.Module):
    """
    This nn module converts input images into embedded tokens with positional encodings
    and includes a learnable class token.
    """
    def __init__(self, image_h=28, image_w=28, image_c=3, patch_d=4, embedding_dim=8, dropout=0.1, device='cuda'):
        super().__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.patch_d = patch_d
        self.embedding_dim = embedding_dim
        self.device = device

        # Define class_token as a learnable parameter with shape [1, 1, embedding_dim]
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        assert image_h % patch_d == 0, f"Image height {image_h} is not divisible by patch size {patch_d}"
        assert image_w % patch_d == 0, f"Image width {image_w} is not divisible by patch size {patch_d}"

        self.N_patches = (image_h * image_w) // patch_d ** 2
        self.max_seq_len = self.N_patches + 1  # +1 for class token

        # Linear layer for patch embedding
        self.patch_embedding_layer = nn.Linear(self.patch_d ** 2 * image_c, self.embedding_dim)

        # Positional embeddings including class token
        self.position_embedding = nn.Embedding(self.max_seq_len, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        """
        Converts an input batch containing images of size cxhxw
        to batches of the tokens in embedded vectorised format
        along with a class token.

        Arguments:
            x (tensor): Input batch of shape (batch_size, image_c, image_h, image_w)
        Returns:
            patch_pos_embedding (tensor): Embedded batch along with position encodings
                                            of shape (batch_size, n_patches+1, embedding_dim)
        """
        assert(x.dim() == 4), "Input batch should have a shape of (B, image_c, image_h, image_w)"
        batch_size = x.shape[0]
        positions = torch.arange(0, self.max_seq_len).expand(batch_size, self.max_seq_len).to(self.device)
        position_encodings = self.position_embedding(positions)

        patches = torch.zeros((batch_size, self.N_patches, self.patch_d**2*self.image_c)).to(device=self.device)
        for i in range(self.image_h//self.patch_d):
            for j in range(self.image_w//self.patch_d):
                patches[:,i*(self.image_w//self.patch_d)+j] = x[:,:,i*self.patch_d:(i+1)*self.patch_d,j*self.patch_d:(j+1)*self.patch_d].reshape(batch_size, self.patch_d**2*self.image_c)

        vector_patches = self.dropout(self.patch_embedding_layer(patches))
        class_tokens = self.class_token.expand(batch_size, 1, self.embedding_dim)
        patch_embedding = torch.hstack([class_tokens, vector_patches])
        patch_pos_embedding = patch_embedding + position_encodings
        patch_pos_embedding = self.dropout(patch_pos_embedding)

        return patch_pos_embedding


def train_loop(epoch, dataloader, model, loss_fn, optimizer, wandb, device='cuda', train_step=0):
    """
    Trains the model for one epoch and logs training loss and accuracy per batch.

    Args:
        epoch (int): Current epoch number.
        dataloader (DataLoader): DataLoader for training data.
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        wandb (wandb.sdk.wandb_run.Run): WandB run for logging.
        device (str): Device to run the training on.
        train_step (int): Current training step count.
    
    Returns:
        float: Average training loss over the epoch.
        float: Average training accuracy over the epoch.
    """
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0.0
    correct = 0

    with tqdm(dataloader, desc=f"Epoch {epoch} [Train]", unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            
            # Calculate predictions and accuracy
            _, preds = torch.max(outputs, dim=1)
            batch_correct = (preds == target).sum().item()
            batch_accuracy = batch_correct / data.size(0)

            # Update metrics
            total_loss += loss.item()
            correct += batch_correct

            # Log per batch with train_step
            wandb.log({
                "Train Loss": loss.item(),
                "Train Accuracy": batch_accuracy * 100,
                "train_step": train_step
            })

            # Increment train_step
            train_step += 1

            # Log per batch
            wandb.log({
                "Train Loss": loss.item(),
                "Train Accuracy": batch_accuracy * 100,
                "train_step": train_step
            })

            # Update progress bar
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * batch_accuracy)

    avg_loss = total_loss / num_batches
    avg_accuracy = (correct / size) * 100
    print(f"Training -- Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.2f}%")
    return avg_loss, avg_accuracy, train_step

def test_loop(epoch, dataloader, model, loss_fn, wandb, device='cuda'):
    """
    Evaluates the model on the test dataset and logs testing loss and accuracy.

    Args:
        epoch (int): Current epoch number.
        dataloader (DataLoader): DataLoader for test data.
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): Loss function.
        wandb (wandb.sdk.wandb_run.Run): WandB run for logging.
        device (str): Device to run the evaluation on.
    
    Returns:
        float: Average testing loss over the epoch.
        float: Testing accuracy over the epoch.
    """
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        with tqdm(dataloader, desc=f"Epoch {epoch} [Test]", unit="batch") as tepoch:
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = loss_fn(outputs, target)
                
                # Calculate predictions and accuracy
                _, preds = torch.max(outputs, dim=1)
                batch_correct = (preds == target).sum().item()
                batch_accuracy = batch_correct / data.size(0)

                # Update metrics
                total_loss += loss.item()
                correct += batch_correct

                # Update progress bar
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * batch_accuracy)

    avg_loss = total_loss / num_batches
    avg_accuracy = (correct / size) * 100
    print(f"Testing -- Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.2f}%")

    # Log per epoch with epoch as the step
    wandb.log({
        "Test Loss": avg_loss,
        "Test Accuracy": avg_accuracy,
        "epoch": epoch  # Use 'epoch' as the step for testing metrics
    })

    return avg_loss, avg_accuracy

def checkpoint(model, filename):
    """
    Saves the model's state dictionary to a file.

    Args:
        model (nn.Module): The neural network model.
        filename (str): Path to save the model.
    """
    torch.save(model.state_dict(), filename)
    print(f"Model checkpoint saved to {filename}")

def resume(model, filename, device='cuda'):
    """
    Loads the model's state dictionary from a file.

    Args:
        model (nn.Module): The neural network model.
        filename (str): Path to load the model from.
        device (str): Device to map the model.
    """
    model.load_state_dict(torch.load(filename, map_location=device))
    model.to(device)
    print(f"Model loaded from {filename}")

if __name__=="__main__":
    from torch.utils.data import Dataset
    from torchvision import datasets
    from torchvision.transforms import ToTensor

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

    trainloader = torch.utils.data.DataLoader(training_data, batch_size=10,
                                          shuffle=True)
    dataiter = iter(trainloader)

    images,labels = next(dataiter)

    print(images.shape)

    patch_embedding = PatchPositionEmbedding(image_h=28, image_w=28, image_c=1, patch_d=4, embedding_dim=8, device='cuda')

    embeddings = patch_embedding(images)

    print(embeddings.shape)