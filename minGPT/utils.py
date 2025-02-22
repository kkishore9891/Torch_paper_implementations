"""
This module contains utility functions required for creating the transformer module.
"""

import math as m
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os

class InputEmbedding(nn.Module):
    """
    Converts input token indices into embedding vectors.
    """
    def __init__(self, vocab_len: int = 10, embedding_dim: int = 512, device: str = 'cuda'):
        """
        Args:
            vocab_len: Size of the vocabulary.
            embedding_dim: Dimensionality of the embedding vectors.
            device: Device for the embedding weights.
        """
        super(InputEmbedding, self).__init__()
        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_len,
                                            embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len) containing token indices.

        Returns:
            Tensor: Embedded representation of shape (batch_size, seq_len, embedding_dim).
        """
        if x.dim() != 2:
            raise ValueError("Input batch should have a shape of (B, N)")
        return self.embedding_layer(x)


def PositionalEncoding(max_seq_len: int = 1000, embedding_dim: int = 768, device: str = 'cuda') -> torch.Tensor:
    """
    Generates positional encoding embeddings for a sequence.

    Args:
        max_seq_len: Maximum sequence length.
        embedding_dim: Dimensionality of the embedding vectors.
        device: Device to which the positional encodings are moved.

    Returns:
        Tensor: Positional encodings of shape (max_seq_len, embedding_dim).
    """
    pos_embeddings = torch.zeros((max_seq_len, embedding_dim))
    for pos in range(max_seq_len):
        for i in range(embedding_dim // 2):
            pos_embeddings[pos, 2 * i] = m.sin(pos / (5 ** (2 * i / embedding_dim)))
            pos_embeddings[pos, 2 * i + 1] = m.cos(pos / (5 ** (2 * i / embedding_dim)))
    return pos_embeddings.to(device=device)

def train_loop(
    epoch,
    dataloader,
    model,
    mask,
    loss_fn,
    optimizer,
    wandb,
    device='cuda',
    train_step=0,
    generate_every=1000
):
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0.0
    correct = 0

    with tqdm(dataloader, desc=f"Epoch {epoch} [Train]", unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch, start=1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward pass with causal mask
            outputs = model(data, target_mask=mask)

            loss = loss_fn(outputs.view(-1, outputs.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()

            # Accuracy for the batch
            _, preds = torch.max(outputs.view(-1, outputs.size(-1)), dim=1)
            batch_correct = (preds == target.view(-1)).sum().item()
            batch_accuracy = batch_correct / target.numel()

            # Update aggregated metrics
            total_loss += loss.item()
            correct += batch_correct

            # Increment step
            train_step += 1

            # Single logging call
            wandb.log({
                "Train Loss": loss.item(),
                "Train Accuracy": batch_accuracy * 100,
                "train_step": train_step
            })

            # Occasionally generate sample
            if train_step % generate_every == 0:
                input_ids = dataloader.dataset.encode_text("Et tu, Brute!")
                input_tensor = torch.tensor(input_ids, dtype=torch.long)[None].to(device)
                prediction = model.generate(input_tensor, max_new_tokens=100)
                print(dataloader.dataset.decode_tokens(prediction[0].cpu().tolist()))

            tepoch.set_postfix(loss=loss.item(), accuracy=100. * batch_accuracy)

    # Compute average loss/accuracy over the epoch
    avg_loss = total_loss / num_batches
    avg_accuracy = (correct / size) * 100

    # Log epoch-level metrics
    # Perplexity = exp(cross_entropy)
    avg_perplexity = m.exp(avg_loss)

    print(f"Training -- Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.2f}%, "
          f"Avg Perplexity: {avg_perplexity:.2f}")

    # Optionally log these as well
    wandb.log({
        "Train Perplexity": avg_perplexity,
        "Avg Train Loss": avg_loss,
        "Avg Train Accuracy": avg_accuracy,
        "epoch": epoch
    })

    return avg_loss, avg_accuracy, train_step

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
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename, map_location=device))
        model.to(device)
        print(f"Model loaded from {filename}.")
    else:
        print(f"{filename} not available.")

class Tokenization(Dataset):
    """
    This class is responsible for tokenizing a large chunk of text into
    tokens by assigning an index to every character present in the text.
    Every unique character in the text is assigned an index and the
    resulting dictionary in turn will be used to convert the characters
    to token indices. One hot encoding is not performed here.
    """
    def __init__(self, text, context_len = 100):
        """
        Function that initialises the Tokenisation class by processing
        the text to calculate the tokenisation parameters

        Args:
            text (str): Raw string that will be tokenised for training.
        """
        unique_chars = sorted(set(text))
        total_tokens, vocab_len = len(text), len(unique_chars)

        print(f"The current dataset consists of {total_tokens} tokens and {vocab_len} unique symbols that can be fed to and predicted by the LLM.")
        
        self.char_encode_dict = {unique_char:idx for idx,unique_char in enumerate(unique_chars)}
        self.token_decode_dict = {idx:unique_char for idx,unique_char in enumerate(unique_chars)}

        self.context_len = context_len
        self.vocab_len = vocab_len
        self.text_dataset = text
        self.total_tokens = total_tokens

    def encode_text(self, text_block):
        tokenized_block = [self.char_encode_dict[char] for char in text_block]

        return tokenized_block
    
    def decode_tokens(self, tokenized_block):
        text_block = [self.token_decode_dict[token] for token in tokenized_block]
        text_block = ''.join(text_block)
        return text_block

    
    def __len__(self):
        """
        Returns the total number of starting tokens from which
        context blocks can be sampled from.
        """
        return self.total_tokens - self.context_len
    
    def __getitem__(self, idx):
        """
        Returns the tokenized characters of context length 'block size'
        from the text dataset's index idx.
        """
        text_block = self.text_dataset[idx:idx+self.context_len+1]
        tokenized_block = self.encode_text(text_block)

        x = torch.tensor(tokenized_block[:self.context_len], dtype = torch.long)
        y = torch.tensor(tokenized_block[1:], dtype = torch.long)

        return x, y

if __name__ == "__main__":
    input_embedding = InputEmbedding(vocab_len=10, embedding_dim=10)
    input_batch = torch.tensor([[0, 1, 2, 3, 4],
                                [5, 6, 7, 8, 9],
                                [9, 8, 7, 6, 5],
                                [4, 3, 2, 1, 0]]).to('cuda')
    print(f"Feeding a tensor of shape {input_batch.shape}")
    embedded_batch = input_embedding(input_batch)
    print(f"The shape of the embedded batch is {embedded_batch.shape}")
    print(f"The embedded batch is {embedded_batch}")

    position_embeddings = PositionalEncoding(max_seq_len=5, embedding_dim=10)
    print(f"The shape of the positional encoding embeddings is: {position_embeddings.shape}")
    print(f"The positional encoding embeddings: {position_embeddings}")
    print((embedded_batch + position_embeddings).shape)
