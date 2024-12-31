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

        self.dot_prod = self.softmax(self.dot_prod)

        # self.attention = torch.einsum("bhqv,bvhd->bqhd", self.dot_prod, self.value).reshape(self.batch_size,self.query_len, self.d_model)
        self.attention = torch.matmul(self.dot_prod, torch.permute(self.value, (0,2,1,3))) 
        self.attention = torch.permute(self.attention, (0,2,1,3)).reshape(self.batch_size,self.query_len, self.d_model)
        self.out = self.linear_out(self.attention)

        return self.out
    
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

        self.feed_forward = nn.Sequential(nn.Linear(embedding_dim, ff_multiplier*embedding_dim), nn.ReLU(), nn.Linear(ff_multiplier*embedding_dim, embedding_dim)).to(device=device)

    def forward(self, input, mask=None):
        """
        The encoder block feed forward function

        Arguments:
            input (tensor): The input tensor batch of shape (batch_size, seq_length, embedding_dim)
        
        Returns:
            output (tensor): The output tensor from the encoder block of shape (batch_size, seq_length, embedding_dim)
        """
        attention = self.mhsa(input, input, input, mask)
        ff_input = self.dropout(self.norm1(input+attention))
        ff_output = self.feed_forward(ff_input)
        output = self.dropout(self.norm2(ff_input+ff_output))

        return output

class DecoderBlock(nn.Module):
    """
    A single transformer decoder block which consists of Masked multi headed attention,
    multi-headed attention feed forward layer and some add & norm skip connection layers.
    """
    def __init__(self, n_heads=5, embedding_dim=10, ff_multiplier=4, dropout=0.1, device='cuda'):
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

        self.mhsa1 = MultiHeadAttention(n_heads=self.n_heads, embedding_dim= self.embedding_dim, device=device)
        self.mhsa2 = MultiHeadAttention(n_heads=self.n_heads, embedding_dim= self.embedding_dim, device=device)
        self.norm1 = nn.LayerNorm(self.embedding_dim).to(device=device)
        self.norm2 = nn.LayerNorm(self.embedding_dim).to(device=device)
        self.norm3 = nn.LayerNorm(self.embedding_dim).to(device=device)
        self.dropout = nn.Dropout(p=self.dropout)

        self.feed_forward = nn.Sequential(nn.Linear(embedding_dim, ff_multiplier*embedding_dim), nn.ReLU(), nn.Linear(ff_multiplier*embedding_dim, embedding_dim)).to(device=device)

    def forward(self, query, encoder_out, src_mask=None, target_mask=None):
        """
        The encoder block feed forward function

        Arguments:
            query (tensor): The query tensor batch of shape (batch_size, seq_length, embedding_dim)
            encoder_out (tensor): The output of the transformer encoder blocks of shape 
                                    (batch_size, seq_length, embedding_dim)
        
        Returns:
            output (tensor): The output tensor from the encoder block of shape (batch_size, seq_length, embedding_dim)
        """
        attention1 = self.mhsa1(query, query, query, target_mask)
        mhsa2_input = self.dropout(self.norm1(query+attention1))
        attention2 = self.mhsa2(mhsa2_input, encoder_out, encoder_out, src_mask)
        ff_input = self.dropout(self.norm2(mhsa2_input+attention2))
        ff_output = self.feed_forward(ff_input)
        output = self.dropout(self.norm3(ff_input+ff_output))

        return output

class EncoderStack(nn.Module):
    """
    Class where multiple encoders are stacked together to which input is provided
    """
    def __init__(self, src_vocab_len=10, max_src_seq_len =5, n_encoders=6,
                  n_heads=5, embedding_dim=10, ff_multiplier=4,
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
        self.src_vocab_len = src_vocab_len
        self.max_src_seq_len = max_src_seq_len
        self.embedding_dim = embedding_dim
        self.n_encoders = n_encoders
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.ff_multiplier = ff_multiplier
        self.device = device
        self.dropout = dropout

        self.encoder_input_embedding = InputEmbedding(vocab_len=src_vocab_len, embedding_dim=embedding_dim)
        self.encoder_position_embeddings = PositionalEncoding(max_seq_len=max_src_seq_len,
                                                               embedding_dim=embedding_dim)
        self.layer_list = nn.ModuleList(EncoderBlock(n_heads, embedding_dim, ff_multiplier,dropout,
                                                       device) for i in range(self.n_encoders))
        
    def forward(self, encoder_input_batch, src_mask = None):
        """
        Performs the feed forward step for the transformer encoder.
        
        Arguments:
            encoder_input_batch (tensor): Tensor containing the source input sequences with
                                            token classes of shape (batch_size, sequence_length)
            src_mask (tensor): A mask which is applied in the Multi-Head attention block of
                                shape (seq_length, seq_length)

        """
        self.encoder_embedded_batch = self.encoder_input_embedding(encoder_input_batch)
        self.encoder_input = self.encoder_embedded_batch + self.encoder_position_embeddings[:self.encoder_embedded_batch.shape[1]]

        for encoder_block in self.layer_list:
            self.encoder_input = encoder_block(self.encoder_input, mask= src_mask)
        self.encoder_output = self.encoder_input

        return self.encoder_output
        
class DecoderStack(nn.Module):
    """
    Class where multiple decoders are stacked together to which input is provided
    """
    def __init__(self, targ_vocab_len=10, max_targ_seq_len=5, n_decoders=6, n_heads=5,
                 embedding_dim=10, ff_multiplier=4,
                 dropout=0.1, device='cuda'):
        """
        Initialises the required components of the decoder stack

        Arguments:
            targ_vocab_len (int): The vocabulary length of the target data.
            max_targ_seq_len (int): Maximum value of the sequence length in the target dataset.
            n_decoders (int): Number of decoder blocks to be stacked together.
            n_heads (int): Number of Self attention blocks per each decoder.
            embedding_dim (int): Length of a token embedding in the sequence.
            ff_multiplier (int): The number of neurons in the hidden layer of the
                                    feed forward block is determined by this.
            dropout (int): Dropout probability.
            device (str): The device to which weights are loaded. Defaults to cuda.
        """
        super().__init__()
        self.targ_vocab_len = targ_vocab_len
        self.max_targ_seq_len = max_targ_seq_len
        self.n_decoders = n_decoders
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.ff_multiplier = ff_multiplier
        self.device = device
        self.dropout = dropout

        self.decoder_input_embedding = InputEmbedding(vocab_len=targ_vocab_len, embedding_dim=embedding_dim)
        self.decoder_position_embeddings = PositionalEncoding(max_seq_len=max_targ_seq_len,
                                                              embedding_dim=embedding_dim)
        self.layer_list = nn.ModuleList(DecoderBlock(n_heads, embedding_dim, ff_multiplier,dropout,
                                                      device) for i in range(n_decoders))
        
    def forward(self, decoder_input_batch, encoder_output, src_mask = None, target_mask=None):
        """
        Performs the feed forward step for the transformer decoder.
        
        Arguments:
            decoder_input_batch (tensor): Tensor containing the target input sequences with
                                            token classes of shape (batch_size, sequence_length)
            src_mask (tensor): A mask which is applied in the Multi-Head attention block of
                                shape (seq_length, seq_length)

        """
        self.decoder_embedded_batch = self.decoder_input_embedding(decoder_input_batch)
        self.decoder_input = self.decoder_embedded_batch + self.decoder_position_embeddings[:self.decoder_embedded_batch.shape[1]]
        self.encoder_output = encoder_output

        for decoder_block in self.layer_list:
            self.decoder_input = decoder_block(self.decoder_input, self.encoder_output,
                                               src_mask= src_mask, target_mask=target_mask)
        self.decoder_output = self.decoder_input

        return self.decoder_output        

class Transformer(nn.Module):
    """
    The full transformer with the encoder and decoder stacks.
    """
    def __init__(self, src_vocab_len=10, targ_vocab_len=10, max_seq_len=5, dropout=0.1, 
                 n_encoders=6, n_decoders=6, n_heads=5, embedding_dim=10, ff_multiplier=4, device='cuda'):
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

        self.encoder_stack = EncoderStack(src_vocab_len=src_vocab_len, max_src_seq_len=max_seq_len,
                                     n_encoders=n_encoders, n_heads=n_heads, embedding_dim=embedding_dim,
                                     ff_multiplier=ff_multiplier, dropout=dropout, device=device)
        self.decoder_stack = DecoderStack(targ_vocab_len=targ_vocab_len, max_targ_seq_len=max_seq_len,
                                     n_decoders=n_decoders, n_heads=n_heads, embedding_dim=embedding_dim,
                                     ff_multiplier=ff_multiplier, dropout=dropout, device=device)
        
        self.prediction_layer = nn.Linear(embedding_dim, targ_vocab_len).to(device=device)
        self.softmax = nn.Softmax(dim=-1)

        
    def forward(self, encoder_input_batch, decoder_input_batch, src_mask=None, targ_mask=None):
        """
        Performs the entire feed forward for the transformer.

        Arguments:
            encoder_input_batch (tensor): Tensor containing the source input sequences with
                                            token classes of shape (batch_size, sequence_length)
            decoder_input_batch (tensor): Tensor containing the target input sequences with
                                            token classes of shape (batch_size, sequence_length)
        Returns:
            predicted_label (tensor): Prediction vector of shape (batch_size, sequence_length, target_vocab_len)
        """
        encoder_output = self.encoder_stack(encoder_input_batch, src_mask)
        decoder_output = self.decoder_stack(decoder_input_batch, encoder_output, src_mask=None, target_mask = targ_mask)
        predicted_label = self.softmax(self.prediction_layer(decoder_output))

        return predicted_label




if __name__ == "__main__":
    # encoder_input_embedding = InputEmbedding(vocab_len=10, embedding_dim=20)
    encoder_input_batch = torch.tensor([[6, 1, 2, 3, 4],[5, 6,7,8,9], [9, 8,7,6,5], [4, 3, 2, 1, 0]]).to('cuda')
    # encoder_embedded_batch = encoder_input_embedding(encoder_input_batch)
    # encoder_position_embeddings = PositionalEncoding(max_seq_len=5, embedding_dim=20)
    
    # decoder_input_embedding = InputEmbedding(vocab_len=10, embedding_dim=10)
    decoder_input_batch = torch.tensor([[0, 1, 1, 2, 6, 7],[7, 2, 9, 3, 6, 5], [3, 1, 7, 9, 4, 1], [4, 2, 6, 5, 8, 2]]).to('cuda')
    # decoder_embedded_batch = decoder_input_embedding(decoder_input_batch)
    # decoder_position_embeddings = PositionalEncoding(max_seq_len=6, embedding_dim=10)
    
    # encoder_input = encoder_embedded_batch + encoder_position_embeddings
    
    # decoder_input = decoder_embedded_batch + decoder_position_embeddings
    mask = torch.triu(torch.ones(6,6),diagonal=1).bool().to('cuda')

    # mhsa = MultiHeadAttention(n_heads=5, embedding_dim=20)
    # output = mhsa(encoder_input, encoder_input, encoder_input, mask=mask)
    # print(encoder_input.shape, output.shape)

    # encoder = EncoderBlock()
    # encoder_output = encoder(encoder_input)
    # print(encoder_input.shape, encoder_output.shape)

    # decoder = DecoderBlock()
    # decoder_output = decoder(decoder_input, encoder_output, target_mask=mask)
    # print(decoder_input.shape, decoder_output.shape)

    # encoder_stack = EncoderStack(n_encoders=6, n_heads=5, embedding_dim=10, ff_multiplier=4, device='cuda')
    # decoder_stack = DecoderStack(n_decoders=6, n_heads=5, embedding_dim=10, ff_multiplier=4, device='cuda')

    # encoder_output = encoder_stack(encoder_input_batch)
    # decoder_output = decoder_stack(decoder_input_batch, encoder_output, target_mask = mask)

    transformer = Transformer(src_vocab_len=10, targ_vocab_len=10, max_seq_len=100,
                              dropout=0.1, n_encoders=6, n_decoders=6, n_heads=5,
                              embedding_dim=20, ff_multiplier=4, device='cuda')
    output = transformer(encoder_input_batch, decoder_input_batch, targ_mask = mask)

    print(output,output.shape)

