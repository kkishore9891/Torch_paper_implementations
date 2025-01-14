import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PatchPositionEmbedding

class MultiHeadAttention(nn.Module):
    """
    Generic Multi-Head self attention module (MHSA) which takes, query, key
    and values tensors along with an optional Mask that can be applied on
    the query key product to mask unwanted values while computing the
    attention vectors for the sequence. Splits the query key values into
    subsets based on the number of heads.
    """
    def __init__(self, n_heads=5, embedding_dim=10, device='cuda'):
        """
        Initialises the nn layers needed for computing the
        self attention values.

        Arguments:
            n_heads (int): Number of heads required in the MHSA module
            embedding_dim (int): Length of an embedded token in the input sequence.
            device (str): The device to which weights are loaded. Defaults to cuda.
        """
        super().__init__()
        self.d_model = embedding_dim
        self.n_heads = n_heads
        # Computing length of the input for an MHSA head
        self.d_q = self.d_k = self.d_v = self.d_model//self.n_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model).to(device)
        self.k_proj = nn.Linear(self.d_model, self.d_model).to(device)
        self.v_proj = nn.Linear(self.d_model, self.d_model).to(device)

        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(self.d_model, self.d_model).to(device)

    def forward(self, Q, K, V, mask=None):
        """
        Calculates the Multi-Head Attention value based on the received Q,K,V and Mask values.

        Arguments:
            Q (Tensor): Query for the MHSA which is of the shape (batch_size, seq_len, embedding_dim)
            K (Tensor): Key for the MHSA which is of the shape (batch_size, seq_len, embedding_dim)
            V (Tensor): Value for the MHSA which is of the shape (batch_size, seq_len, embedding_dim)

        Returns:
            out (tensor): Output of the MHSA block of shape (batch_size, seq_len, embedding_dim)s
        """

        assert(Q.shape[-1]%self.n_heads == 0), f"Query token length {Q.shape[-1]} is not divisible by the number of heads {self.n_heads}"
        assert(K.shape[-1]%self.n_heads == 0), f"Key token length {K.shape[-1]} is not divisible by the number of heads {self.n_heads}"
        assert(V.shape[-1]%self.n_heads == 0), f"Value token length {V.shape[-1]} is not divisible by the number of heads {self.n_heads}"
        
        # Linearly projecting the Query, Key and Value
        self.query = self.q_proj(Q)
        self.key = self.k_proj(K)
        self.value = self.v_proj(V)
        self.mask = mask

        self.batch_size = self.query.shape[0]
        self.query_len = self.query.shape[1]
        self.key_len = self.key.shape[1]
        self.value_len = self.value.shape[1]

        # assert(self.query.shape[-1] == self.d_model), f"Query token should be of the length {self.d_model}"
        # assert(self.key.shape[-1] == self.d_model), f"Key token should be of the length {self.d_model}"
        # assert(self.value.shape[-1] == self.d_model), f"Value token should be of the length {self.d_model}"

        self.query = self.query.reshape(self.batch_size, self.query_len, self.n_heads, self.d_q) #B, N, H, d_q
        self.key = self.key.reshape(self.batch_size, self.key_len, self.n_heads, self.d_k) #B, N, H, d_k
        self.value = self.value.reshape(self.batch_size, self.value_len, self.n_heads, self.d_v) #B, N, H, d_v

        # self.dot_prod = torch.einsum("bqhd,bkhd->bhqk",self.query, self.key)
        self.dot_prod = torch.matmul(torch.permute(self.query, (0,2,1,3)), torch.permute(self.key, (0,2,3,1)))

        if not self.mask == None:
            mask = mask.bool()
            self. dot_prod = self.dot_prod.masked_fill(mask, value=-1*torch.inf)

        self.dot_prod = self.softmax(self.dot_prod/(self.d_model**(1/2)))

        # self.attention = torch.einsum("bhqv,bvhd->bqhd", self.dot_prod, self.value).reshape(self.batch_size,self.query_len, self.d_model)
        self.attention = torch.matmul(self.dot_prod, torch.permute(self.value, (0,2,1,3))) 
        self.attention = torch.permute(self.attention, (0,2,1,3)).reshape(self.batch_size,self.query_len, self.d_model)
        out = self.linear_out(self.attention)

        return out
    
class EncoderBlock(nn.Module):
    """
    A single transformer encoder block which consists of Multi headed attention,
    feed forward layer and some add & norm skip connection layers.
    """
    def __init__(self, n_heads=5, embedding_dim=10, ff_multiplier=4, dropout = 0.1, device='cuda'):
        """
        Initialises the functional blocks present in the Transformer encoder.

        Arguments:
            n_heads (int): Number of transformer heads required.
            embedding_dim (int): Length of a token embedding in the sequence.
            ff_multiplier (int): The number of neurons in the hidden layer of the
                                    feed forward block is determined by this.
            dropout (int): Dropout probability.
            device (str): The device to which weights are loaded. Defaults to cuda.
        """
        super().__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.ff_multiplier = ff_multiplier
        self.dropout = dropout

        self.mhsa = MultiHeadAttention(n_heads=n_heads, embedding_dim= embedding_dim, device=device)
        self.norm1 = nn.LayerNorm(embedding_dim).to(device=device)
        self.norm2 = nn.LayerNorm(embedding_dim).to(device=device)
        self.dropout = nn.Dropout(p=dropout)

        self.feed_forward = nn.Sequential(nn.Linear(embedding_dim, ff_multiplier*embedding_dim), nn.GELU(), nn.Linear(ff_multiplier*embedding_dim, embedding_dim)).to(device=device)

    def forward(self, input, mask=None):
        """
        The encoder block feed forward function

        Arguments:
            input (tensor): The input tensor batch of shape (batch_size, seq_length, embedding_dim)
        
        Returns:
            output (tensor): The output tensor from the encoder block of shape (batch_size, seq_length, embedding_dim)
        """
        norm_input = self.norm1(input)
        attention = self.dropout(self.mhsa(norm_input, norm_input, norm_input, mask))
        attention = attention + input
        norm_attention = self.norm2(attention)
        ff_output = self.dropout(self.feed_forward(norm_attention))
        output = attention+ff_output

        return output

class EncoderStack(nn.Module):
    """
    Class where multiple encoders are stacked together to which input is provided
    """
    def __init__(self, image_h=28, image_w=28, image_c=28, patch_d=4, n_encoders=6,
                  n_heads=2, embedding_dim=8, ff_multiplier=4,
                   dropout=0.1, device='cuda'):
        """
        Initialises the required components of the encoder stack

        Arguments:
            src_vocab_len (int): The vocabulary length of the target data.
            max_src_seq_len (int): Maximum value of the sequence length in the source dataset.
            n_encoder (int): Number of encoder blocks to be stacked together.
            n_heads (int): Number of Self attention blocks per each encoder.
            embedding_dim (int): Length of a token embedding in the sequence.
            ff_multiplier (int): The number of neurons in the hidden layer of the
                                    feed forward block is determined by this.
            dropout (int): Dropout probability.
            device (str): The device to which weights are loaded. Defaults to cuda.
        """
        super().__init__()
        
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.patch_d = patch_d

        self.embedding_dim = embedding_dim
        self.n_encoders = n_encoders
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.ff_multiplier = ff_multiplier
        self.device = device
        self.dropout = dropout

        self.patch_embedding = PatchPositionEmbedding(image_h=image_h, image_w=image_w, image_c=image_c,
                                                      patch_d=patch_d, embedding_dim=embedding_dim, device='cuda')

        self.layer_list = nn.ModuleList(EncoderBlock(n_heads, embedding_dim, ff_multiplier,dropout,
                                                       device) for i in range(self.n_encoders))
        
    def forward(self, images):
        """
        Performs the feed forward step for the transformer encoder.
        
        Arguments:
            images (tensor): Tensor containing a batch of input images 
                                of shape (batch_size, image_c, image_h, image_w)

        Returns:
            encoder_output (tensor): The output of the transformer encoder stack of shape 
                                    (batch_size, n_patches+1, embedding_dim)

        """
        
        self.encoder_input = self.patch_embedding(images)

        for i,encoder_block in enumerate(self.layer_list):
            self.encoder_input = encoder_block(self.encoder_input)
        self.encoder_output = self.encoder_input

        return self.encoder_output
        
class VisionTransformer(nn.Module):
    """
    The full transformer with the encoder and decoder stacks.
    """
    def __init__(self, image_h=28, image_w=28, image_c=28, patch_d=4, dropout=0.1, 
                 n_encoders=6, n_heads=2, embedding_dim=8, ff_multiplier=4,n_classes=10, device='cuda'):
        """
        Initialises the encoder and decoder stack.
        Arguments:
            src_vocab_len (int): The vocabulary length of the target data.
            targ_vocab_len (int): The vocabulary length of the target data.
            max_seq_len (int): Maximum value of the sequence length in the source and target datasets.
            dropout (int): Dropout probability.
            n_encoder (int): Number of encoders to be stacked.
            n_decoder (int): Number of decoders to be stacked.
            n_heads (int): Number of Self attention blocks per each decoder.
            embedding_dim (int): Length of a token embedding in the sequence.
            ff_multiplier (int): The number of neurons in the hidden layer of the
                                    feed forward block is determined by this.
            device (str): The device to which weights are loaded. Defaults to cuda.
        """
        super().__init__()

        self.encoder_stack = EncoderStack(image_h=image_h, image_w=image_w, image_c=image_c,
                                          patch_d=patch_d, n_encoders=n_encoders,
                                          n_heads=n_heads, embedding_dim=embedding_dim,
                                          ff_multiplier=ff_multiplier, dropout=dropout, device='cuda')
        
        self.prediction_layer = nn.Sequential(nn.Linear(embedding_dim, n_classes), nn.LogSoftmax(dim=1)).to(device=device)

        
    def forward(self, images):
        """
        Performs the entire feed forward for the transformer.

        Arguments:
            images (tensor): Tensor containing a batch of input images 
                                of shape (batch_size, image_c, image_h, image_w)
        Returns:
            predicted_label (tensor): Prediction vector of shape (batch_size, sequence_length, target_vocab_len)
        """
        encoder_output = self.encoder_stack(images)

        predicted_label = self.prediction_layer(encoder_output[:,0])

        return predicted_label
