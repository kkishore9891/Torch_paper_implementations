import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
import wandb
import datetime
from model import GPT
from utils import Tokenization
from utils import train_loop, checkpoint, resume
import wandb
import datetime

input_text = open('shakespeare.txt', 'r').read()
context_len = 128

train_dataset = Tokenization(input_text, context_len=context_len)
vocab_len = train_dataset.vocab_len
model = GPT(src_vocab_len=vocab_len, max_seq_len=context_len,
            dropout=0.1, n_decoders=6, n_heads=6,
            embedding_dim=192, ff_multiplier=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resume(model=model, filename="models/shakespeare/shakespeare_best.pth")

model = model.to(device=device)

summary(model)

# Hyperparameters
learning_rate = 1e-4
batch_size = 512
epochs = 30

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize wandb
best_train_accuracy = 0
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
wandb.init(
    project="GPT_shakespeare",
    name=f"minGPT implementation_exp_{curr_time}",
    config={
        "learning_rate": learning_rate,
        "architecture": "GPT-2",
        "dataset": "shakespeare.txt",
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss"
    }
)

# Define separate step metrics for training and testing
wandb.define_metric("train_step")
wandb.define_metric("Train Loss", step_metric="train_step")
wandb.define_metric("Train Accuracy", step_metric="train_step")

wandb.define_metric("epoch")
wandb.define_metric("Test Loss", step_metric="epoch")
wandb.define_metric("Test Accuracy", step_metric="epoch")

# Initialize train_step counter
train_step = 0

# Log model architecture
wandb.watch(model, log="all")

causal_mask = torch.triu(
                torch.ones(context_len, context_len, dtype=torch.bool),
                diagonal=1
            ).to(device)

# Training Loop
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}/{epochs}\n{'-'*30}")

    # Training Phase
    train_loss, train_accuracy, train_step = train_loop(
        epoch=epoch,
        dataloader=train_loader,
        model=model,
        mask = causal_mask,
        loss_fn=loss_fn,
        optimizer=optimizer,
        wandb=wandb,
        device=device,
        train_step=train_step
    )
    
    # Checkpointing
    if train_accuracy > best_train_accuracy:
        print(f"New best accuracy: {train_accuracy:.2f}% (previous: {best_train_accuracy:.2f}%)")
        best_train_accuracy = train_accuracy
        checkpoint(model=model, filename=f"models/shakespeare/shakespeare_{curr_time}_best.pth")
    
    print("\n")

print("Training Complete!")
wandb.finish()

