"""
This program contains useful functions required while creating the transformer module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m

class InputEmbedding(nn.Module):
    """
    This is an nn module that takes input token class numbers and converts them
    to an embedding vector.
    """
    def __init__(self, vocab_len = 10, embedding_dim = 512, device = 'cuda'):
        """
        Init function that initialises the embedding layer.

        Arguments:
            vocab_len (int): The vocabulary length of the input data.
            embedding_dim (int): The size of the embedded vector for each token.
            device (str): The device to which weights are loaded. Defaults to cuda.
        """
        super().__init__()
        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_len,
                                             embedding_dim=embedding_dim).to(device=device)

    def forward(self, x):
        """
        Converts an input batch containing sequences with token class numbers
        to batches of the tokens in embedded vectorised format

        Arguments:
            x (tensor): Input batch of shape (batch_size, seq_len)
        Returns:
            embedding (tensor): Embedded batch of shape (batch_size, seq_len, embedding_dim)
        """
        assert(x.dim() == 2), "Input batch should have a shape of (B,N)"
        return self.embedding_layer(x)

def PositionalEncoding(max_seq_len = 5, embedding_dim = 512, device = 'cuda'):
    """
    Creates the position encoder embeddings which will be added to the input
    sequences embeddings to keep track of positions of the sequence tokens.

    Arguments:
        max_seq_len (int): Maximum value of the sequence length in the dataset.
        embedding_dim (int): The length of the embedding vector for a sequence token.
        device (str): The device to which weights are loaded. Defaults to cuda.
    Returns:
        pos_embeddings (tensor): Position embeddings vector of dimension (max_seq_len, embedding_dim)
    """
    
    pos_embeddings = torch.zeros((max_seq_len, embedding_dim))
    for pos in range(max_seq_len):
        for i in range(embedding_dim//2):
            pos_embeddings[pos, 2*i] = m.sin(pos/(5**(2*i/embedding_dim)))
            pos_embeddings[pos, 2*i+1] = m.cos(pos/(5**(2*i/embedding_dim)))
    
    return pos_embeddings.to(device=device)


if __name__=="__main__":
    input_embedding = InputEmbedding(vocab_len=10, embedding_dim=10)
    
    # input_batch = torch.tensor([[[0, 1, 2, 3, 4],[5, 6,7,8,9], [9, 8,7,6,5], [4, 3, 2, 1, 0]]])
    # print(f"Feeding a tensor of shape {input_batch.shape}")
    # embedded_batch = input_embedding(input_batch)
    # print(f"The shape of the embedded batch is {embedded_batch.shape}")

    input_batch = torch.tensor([[0, 1, 2, 3, 4],[5, 6,7,8,9], [9, 8,7,6,5], [4, 3, 2, 1, 0]]).to('cuda')
    print(f"Feeding a tensor of shape {input_batch.shape}")
    embedded_batch = input_embedding(input_batch)
    print(f"The shape of the embedded batch is {embedded_batch.shape}")
    print(f"The embedded batch is {embedded_batch}")

    position_embeddings = PositionalEncoding(max_seq_len=5, embedding_dim=10)

    print(f"The shape of the positional encoding embeddings is: {position_embeddings.shape}")
    print(f"The positional encoding embeddings is: {position_embeddings}")

    print((embedded_batch + position_embeddings).shape)