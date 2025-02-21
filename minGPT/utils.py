"""
This module contains utility functions required for creating the transformer module.
"""

import math as m
import torch
import torch.nn as nn
from torch.utils.data import Dataset

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


class Tokenization(Dataset):
    """
    This class is responsible for tokenizing a large chunk of text into
    tokens by assigning an index to every character present in the text.
    Every unique character in the text is assigned an index and the
    resulting dictionary in turn will be used to convert the characters
    to token indices. One hot encoding is not performed here.
    """
    def __init__(self, text):
        """
        Function that initialises the Tokenisation class by processing
        the text to calculate the tokenisation parameters

        Args:
            text (str): Raw string that will be tokenised for training.
        """
        unique_chars = sorted(set(text))
        total_tokens, vocab_len = len(text), len(unique_chars)

        print(f"The current dataset consists of {total_tokens} tokens
               and {vocab_len} unique symbols that can be fed to and
               predicted by the LLM.")
        
        self.char_encode_dict = {unique_char:idx for idx,unique_char in enumerate(unique_chars)}
        self.token_decode_dict = {idx:unique_char for idx,unique_char in enumerate(unique_chars)}

        self.context_len = 1000
        self.vocab_len = vocab_len
        self.text_dataset = text
        self.total_tokens = total_tokens
    
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
        tokenized_block = [self.char_encode_dict[char] for char in text_block]

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
