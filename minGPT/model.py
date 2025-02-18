import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import InputEmbedding, PositionalEncoding

class MultiHeadAttention(nn.Module):
    """
    Generic Multi-Head self attention module (MHSA) which takes, query, key
    and values tensors along with an optional Mask that can be applied on
    the query key product to mask unwanted values while computing the
    attention vectors for the sequence. Splits the query key values into
    subsets based on the number of heads.
    """
    def __init__(self, n_heads=5, embedding_dim=10, attn_dropout = 0.1, res_dropout = 0.1, device='cuda'):
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

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_dropout = nn.Dropout(res_dropout)

        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(self.d_model, self.d_model).to(device)

    def forward(self, Q, K, V, mask=None):
        """
        Calculates the Multi-Head Attention value based on the received Q,K,V and Mask values.

        Arguments:
            Q (Tensor): Query for the MHSA which is of the shape (batch_size, seq_len, embedding_dim)
            K (Tensor): Key for the MHSA which is of the shape (batch_size, seq_len, embedding_dim)
            V (Tensor): Value for the MHSA which is of the shape (batch_size, seq_len, embedding_dim)
        """

        assert(Q.shape[-1]%self.n_heads == 0), f"Query token length {Q.shape[-1]} is not divisible by the number of heads {self.n_heads}"
        assert(K.shape[-1]%self.n_heads == 0), f"Key token length {K.shape[-1]} is not divisible by the number of heads {self.n_heads}"
        assert(V.shape[-1]%self.n_heads == 0), f"Value token length {V.shape[-1]} is not divisible by the number of heads {self.n_heads}"
        
        # Linearly projecting the Query, Key and Value
        self.query = self.q_proj(Q)
        self.key = self.k_proj(K)
        self.value = self.v_proj(V)

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
            self.mask = mask[:self.query_len, :self.query_len]
            mask = mask.bool()
            self. dot_prod = self.dot_prod.masked_fill(mask, value=-1*torch.inf)

        self.dot_prod = self.attn_dropout(self.softmax(self.dot_prod/(self.d_k**(1/2))))

        # self.attention = torch.einsum("bhqv,bvhd->bqhd", self.dot_prod, self.value).reshape(self.batch_size,self.query_len, self.d_model)
        self.attention = torch.matmul(self.dot_prod, torch.permute(self.value, (0,2,1,3))) 
        self.attention = torch.permute(self.attention, (0,2,1,3)).reshape(self.batch_size,self.query_len, self.d_model)
        self.out = self.res_dropout(self.linear_out(self.attention))

        return self.out
    
class DecoderBlock(nn.Module):
    """
    A single transformer decoder block which consists of Multi headed attention,
    feed forward layer and some add & norm skip connection layers.
    """
    def __init__(self, n_heads=5, embedding_dim=10, ff_multiplier=4, dropout = 0.1, device='cuda'):
        """
        Initialises the functional blocks present in the Transformer decoder.

        Arguments:
            n_heads (int): Number of attention heads required.
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
        The decoder block feed forward function

        Arguments:
            input (tensor): The input tensor batch of shape (batch_size, seq_length, embedding_dim)
        
        Returns:
            output (tensor): The output tensor from the decoder block of shape (batch_size, seq_length, embedding_dim)
        """
        norm_input = self.norm1(input)
        attention = self.mhsa(norm_input, norm_input, norm_input, mask)
        attention = attention + input
        norm_attention = self.norm2(attention)
        ff_output = self.dropout(self.feed_forward(norm_attention))
        output = attention+ff_output

        return output


class DecoderStack(nn.Module):
    """
    Class where multiple decoders are stacked together to which input is provided
    """
    def __init__(self, src_vocab_len=10, max_src_seq_len =5, n_decoders=6,
                  n_heads=5, embedding_dim=10, ff_multiplier=4,
                   dropout=0.1, device='cuda'):
        """
        Initialises the required components of the decoder stack

        Arguments:
            src_vocab_len (int): The vocabulary length of the target data.
            max_src_seq_len (int): Maximum value of the sequence length in the source dataset.
            n_decoder (int): Number of decoder blocks to be stacked together.
            n_heads (int): Number of Self attention blocks per each decoder.
            embedding_dim (int): Length of a token embedding in the sequence.
            ff_multiplier (int): The number of neurons in the hidden layer of the
                                    feed forward block is determined by this.
            dropout (int): Dropout probability.
            device (str): The device to which weights are loaded. Defaults to cuda.
        """
        super().__init__()
        self.src_vocab_len = src_vocab_len
        self.max_src_seq_len = max_src_seq_len
        self.embedding_dim = embedding_dim
        self.n_decoders = n_decoders
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.ff_multiplier = ff_multiplier
        self.device = device
        self.dropout = dropout

        self.decoder_input_embedding = InputEmbedding(vocab_len=src_vocab_len, embedding_dim=embedding_dim)
        self.decoder_position_embeddings = PositionalEncoding(max_seq_len=max_src_seq_len,
                                                               embedding_dim=embedding_dim)
        self.layer_list = nn.ModuleList(DecoderBlock(n_heads, embedding_dim, ff_multiplier,dropout,
                                                       device) for i in range(self.n_decoders))

        
    def forward(self, decoder_input_batch, target_mask = None):
        """
        Performs the feed forward step for the transformer decoder.
        
        Arguments:
            decoder_input_batch (tensor): Tensor containing the source input sequences with
                                            token classes of shape (batch_size, sequence_length)
            src_mask (tensor): A mask which is applied in the Multi-Head attention block of
                                shape (seq_length, seq_length)
        Returns:
            decoder_output (tensor): The output of the transformer decoder stack of shape 
                                    (batch_size, seq_length, embedding_dim)

        """
        self.decoder_embedded_batch = self.decoder_input_embedding(decoder_input_batch)
        self.decoder_input = self.dropout(self.decoder_embedded_batch + self.decoder_position_embeddings[:self.decoder_embedded_batch.shape[1]])

        for i,decoder_block in enumerate(self.layer_list):
            self.decoder_input = decoder_block(self.decoder_input, mask=target_mask)
        self.decoder_output = self.decoder_input

        return self.decoder_output
               

class GPT(nn.Module):
    """
    The full transformer with the decoder stacks.
    """
    def __init__(self, src_vocab_len=10, targ_vocab_len=10, max_seq_len=5, dropout=0.1, 
                 n_decoders=6, n_heads=5, embedding_dim=10, ff_multiplier=4, device='cuda'):
        """
        Initialises the decoder stack.
        Arguments:
            src_vocab_len (int): The vocabulary length of the target data.
            targ_vocab_len (int): The vocabulary length of the target data.
            max_seq_len (int): Maximum value of the sequence length in the source and target datasets.
            dropout (int): Dropout probability.
            n_decoder (int): Number of decoders to be stacked.
            n_heads (int): Number of Self attention blocks per each decoder.
            embedding_dim (int): Length of a token embedding in the sequence.
            ff_multiplier (int): The number of neurons in the hidden layer of the
                                    feed forward block is determined by this.
            device (str): The device to which weights are loaded. Defaults to cuda.
        """
        super().__init__()

        self.decoder_stack = DecoderStack(src_vocab_len=src_vocab_len, max_src_seq_len=max_seq_len,
                                     n_decoders=n_decoders, n_heads=n_heads, embedding_dim=embedding_dim,
                                     ff_multiplier=ff_multiplier, dropout=dropout, device=device)
        
        self.norm = nn.LayerNorm(embedding_dim)
        self.prediction_layer = nn.Linear(embedding_dim, targ_vocab_len)
        self.softmax = nn.Softmax(dim=-1)

        
    def forward(self, decoder_input_batch, target_mask=None):
        """
        Performs the entire feed forward for the transformer.

        Arguments:
            decoder_input_batch (tensor): Tensor containing the source input sequences with
                                            token classes of shape (batch_size, sequence_length)
        Returns:
            predicted_label (tensor): Prediction vector of shape (batch_size, sequence_length, target_vocab_len)
        """
        decoder_output = self.decoder_stack(decoder_input_batch, target_mask)
        logits = self.prediction_layer(self.norm(decoder_output))

        return logits




if __name__ == "__main__":
    decoder_input_batch = torch.tensor([[6, 1, 2, 3, 4],[5, 6,7,8,9], [9, 8,7,6,5], [4, 3, 2, 1, 0]]).to('cuda')
    max_seq_len=1000
    src_len_vocab = target_len_vocab = 10
    dropout = 0.1
    n_decoders = 6
    n_heads = 5
    embedding_dim = 20
    ff_multiplier=4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mask = torch.triu(torch.ones(max_seq_len,max_seq_len),diagonal=1).bool().to(device)

    gpt = GPT(src_vocab_len=src_len_vocab, targ_vocab_len=target_len_vocab, max_seq_len=max_seq_len,
                              dropout=dropout, n_decoders=n_decoders, n_heads=5,
                              embedding_dim=embedding_dim, ff_multiplier=ff_multiplier, device='cuda')
    output = gpt(decoder_input_batch, target_mask = mask)

    print(output,output.shape)

