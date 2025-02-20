"""
This module contains utility functions required for creating the transformer module.
"""

import math as m
import torch
import torch.nn as nn


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
