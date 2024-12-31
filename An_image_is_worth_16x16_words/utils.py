"""
This program contains useful functions required while creating the transformer module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m

class PatchPositionEmbedding(nn.Module):
    """
    This is an nn module that takes input token class numbers and converts them
    to an embedding vector.
    """
    def __init__(self, image_h = 28, image_w = 28, image_c =1, patch_d=4, embedding_dim = 8, dropout = 0.1, device = 'cuda'):
        """
        Init function that initialises the embedding layer.

        Arguments:
            image_h (int): The height of the image.
            image_w (int): The width of the image.
            image_c (int): The number of channels in the image.
            patch_d (int): The width and height of an image patch.
            embedding_dim (int): The size of the embedded vector for each patch token.
            dropout (int): Dropout probability
            device (str): The device to which weights are loaded. Defaults to cuda.
        """
        super().__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.patch_d = patch_d
        self.embedding_dim = embedding_dim
        self.device = device

        self.class_token = nn.parameter.Parameter(torch.randn(embedding_dim)).to(device=device)
        

        assert(image_h%patch_d==0), f"Image height {image_h} is not divisible by patch size {patch_d}"
        assert(image_w%patch_d==0), f"Image width {image_w} is not divisible by patch size {patch_d}"

        self.N_patches = (image_h*image_w)//patch_d**2
        self.embedding_layer = nn.Linear(self.patch_d**2*image_c, self.embedding_dim).to(device=device)

        self.position_encodings = PositionalEncoding(max_seq_len=self.N_patches+1,
                                                     embedding_dim=self.embedding_dim, device=self.device)
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
        self.batch_size = x.shape[0]

        self.patches = torch.zeros((self.batch_size, self.N_patches, self.patch_d**2*self.image_c)).to(device=self.device)
        for i in range(self.image_h//self.patch_d):
            for j in range(self.image_w//self.patch_d):
                self.patches[:,i*(self.image_w//self.patch_d)+j] = x[:,:,i*self.patch_d:(i+1)*self.patch_d,j*self.patch_d:(j+1)*self.patch_d].reshape(self.batch_size, self.patch_d**2*self.image_c)

        self.patches = self.dropout(self.embedding_layer(self.patches))
        patch_embedding = torch.hstack([self.class_token.expand(self.batch_size,1,self.embedding_dim),self.patches])
        patch_pos_embedding = patch_embedding + self.position_encodings
        patch_pos_embedding = self.dropout(patch_pos_embedding)

        return patch_pos_embedding

def PositionalEncoding(max_seq_len = 50, embedding_dim = 8, device = 'cuda'):
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