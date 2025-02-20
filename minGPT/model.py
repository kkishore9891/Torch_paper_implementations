import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils import InputEmbedding, PositionalEncoding

class MultiHeadAttention(nn.Module):
    """
    Generic Multi-Head self attention module (MHSA) which takes, query, key
    and values tensors along with an optional Mask that can be applied on
    the query key product to mask unwanted values while computing the
    attention vectors for the sequence. Splits the query key values into
    subsets based on the number of heads.
    """
    def __init__(self, n_heads=5, embedding_dim=10, attn_dropout=0.1, res_dropout=0.1):
        """
        Initialises the nn layers needed for computing the
        self attention values.

        Arguments:
            n_heads (int): Number of heads required in the MHSA module
            embedding_dim (int): Length of an embedded token in the input sequence.
            attn_dropout (float): Dropout applied to the softmax applied attention scores.
            res_dropout (float): Dropout applied to the output of the MHSA block.
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_model = embedding_dim
        assert embedding_dim % n_heads == 0, (
            f"Embedding dim ({embedding_dim}) must be divisible by the number of heads ({n_heads})."
        )
        self.d_q = self.d_k = self.d_v = embedding_dim // n_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_dropout = nn.Dropout(res_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, Q, K, V, mask=None):
        """
        Calculates the Multi-Head Attention value based on the received Q,K,V and Mask values.
        
        Args:
            Q, K, V (Tensor): Query, Key and Value tensors of shape (batch_size, seq_len, embedding_dim).
            mask (Tensor, optional): Boolean mask of shape (seq_len, seq_len).

        Returns:
            Tensor: The output of the multi-head attention layer.
        """
        batch_size, query_len, _ = Q.shape
        key_len = K.shape[1]

        # Linear projections and reshape for multiple heads
        query = self.q_proj(Q).reshape(batch_size, query_len, self.n_heads, self.d_q).permute(0, 2, 1, 3)
        key = self.k_proj(K).reshape(batch_size, key_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        value = self.v_proj(V).reshape(batch_size, key_len, self.n_heads, self.d_v).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            # Assume mask is broadcastable to (query_len, key_len)
            scores = scores.masked_fill(mask[:query_len, :query_len], float('-inf'))
        attn = self.attn_dropout(self.softmax(scores))

        # Combine attention output
        context = torch.matmul(attn, value)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, query_len, self.d_model)
        output = self.res_dropout(self.linear_out(context))
        return output


class DecoderBlock(nn.Module):
    """
    A single Transformer decoder block that applies self-attention,
    residual connections, layer normalization, and a feed-forward network.
    """
    def __init__(self, n_heads=5, embedding_dim=10, ff_multiplier=4, dropout=0.1):
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
        self.mhsa = MultiHeadAttention(n_heads=n_heads, embedding_dim=embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_multiplier * embedding_dim),
            nn.GELU(),
            nn.Linear(ff_multiplier * embedding_dim, embedding_dim)
        )

    def forward(self, x, mask=None):
        """
        The decoder block feed forward function

        Arguments:
            x (tensor): The input tensor batch of shape (batch_size, seq_length, embedding_dim)
        
        Returns:
            x (tensor): The output tensor from the decoder block of shape (batch_size, seq_length, embedding_dim)
        """
        # Multi-head self-attention with residual connection
        attn = self.mhsa(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + attn

        # Feed-forward network with residual connection
        ff = self.dropout(self.feed_forward(self.norm2(x)))
        x = x + ff
        return x


class DecoderStack(nn.Module):
    """
    Stacks multiple decoder blocks along with input embeddings and positional encodings.
    """
    def __init__(self, src_vocab_len=10, max_seq_len=5, n_decoders=6,
                 n_heads=5, embedding_dim=10, ff_multiplier=4, dropout=0.1):
        """
        Initialises the required components of the decoder stack

        Arguments:
            src_vocab_len (int): The vocabulary length of the target data.
            max_seq_len (int): Maximum value of the sequence length in the source dataset.
            n_decoder (int): Number of decoder blocks to be stacked together.
            n_heads (int): Number of Self attention blocks per each decoder.
            embedding_dim (int): Length of a token embedding in the sequence.
            ff_multiplier (int): The number of neurons in the hidden layer of the
                                    feed forward block is determined by this.
            dropout (int): Dropout probability.
        """
        super().__init__()
        self.decoder_input_embedding = InputEmbedding(vocab_len=src_vocab_len, embedding_dim=embedding_dim)
        self.decoder_position_embeddings = PositionalEncoding(max_seq_len=max_seq_len,
                                                              embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_list = nn.ModuleList([
            DecoderBlock(n_heads=n_heads, embedding_dim=embedding_dim, ff_multiplier=ff_multiplier, dropout=dropout)
            for _ in range(n_decoders)
        ])

    def forward(self, decoder_input_batch, target_mask=None):
        """
        Args:
            decoder_input_batch (Tensor): Shape (batch_size, seq_len) of token indices.
            target_mask (Tensor, optional): Mask to apply in attention layers.

        Returns:
            Tensor: Decoder stack output of shape (batch_size, seq_len, embedding_dim).
        """
        x = self.decoder_input_embedding(decoder_input_batch)
        # Add positional encoding and apply dropout
        x = x + self.decoder_position_embeddings[:x.shape[1]]
        x = self.dropout(x)

        for block in self.layer_list:
            x = block(x, mask=target_mask)
        return x


class GPT(nn.Module):
    """
    A Transformer-based language model using a stacked decoder architecture.
    """
    def __init__(self, src_vocab_len=10, targ_vocab_len=10, max_seq_len=5, dropout=0.1,
                 n_decoders=6, n_heads=5, embedding_dim=10, ff_multiplier=4):
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
        """
        super().__init__()
        self.decoder_stack = DecoderStack(
            src_vocab_len=src_vocab_len,
            max_seq_len=max_seq_len,
            n_decoders=n_decoders,
            n_heads=n_heads,
            embedding_dim=embedding_dim,
            ff_multiplier=ff_multiplier,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.prediction_layer = nn.Linear(embedding_dim, targ_vocab_len)
        self.softmax = nn.Softmax(dim=-1)
        self.block_size = max_seq_len

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices x (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        # Inspired from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

        for _ in range(max_new_tokens):
            # Sequences  exceed the context length should be cropped to stay within the context length.
            if x.size(1) > self.block_size:
                x = x[:, -self.block_size:]
            
            logits = self(x)

            # Scaling the logits of the final predicted token by temperature to increase, decrease or preserve the randomness of token prediction
            logits = logits[:,-1,:]/temperature

            # Picking topk options to crop the logits.
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # To perform probability distribution based sampling instead of picking top option
            if do_sample:
                x_next = torch.multinomial(probs, num_samples=1)
            else:
                _, x_next = torch.topk(probs, k=1, dim=-1)
            
            x = torch.cat((x, x_next), dim=1)
        
        return x



            
    
    def forward(self, decoder_input_batch, target_mask=None):
        """
        Performs the entire feed forward for the transformer.
        Args:
            decoder_input_batch (Tensor): Shape (batch_size, seq_len) of token indices.
            target_mask (Tensor, optional): Mask applied in attention layers.

        Returns:
            Tensor: Logits of shape (batch_size, seq_len, targ_vocab_len).
        """
        _, seq_len = decoder_input_batch.size()
        assert seq_len <= self.block_size, f"Input exceeds context length!!!"
        decoder_output = self.decoder_stack(decoder_input_batch, target_mask)
        logits = self.prediction_layer(self.norm(decoder_output))
        return logits


if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab_len = 10
    targ_vocab_len = 10
    max_seq_len = 1000
    dropout = 0.1
    n_decoders = 12
    n_heads = 12
    embedding_dim = 768
    ff_multiplier = 4

    # Sample input (batch_size, seq_len)
    decoder_input_batch = torch.tensor(
        [[6, 1, 2, 3, 4],
         [5, 6, 7, 8, 9],
         [9, 8, 7, 6, 5],
         [4, 3, 2, 1, 0]], dtype=torch.long, device=device)

    # Create an upper-triangular mask for causal self-attention
    mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1).to(device)

    # Instantiate and move the model to the device
    gpt = GPT(src_vocab_len=src_vocab_len, targ_vocab_len=targ_vocab_len, max_seq_len=max_seq_len,
              dropout=dropout, n_decoders=n_decoders, n_heads=n_heads,
              embedding_dim=embedding_dim, ff_multiplier=ff_multiplier)
    gpt.to(device)
    summary(gpt)
    # Forward pass
    output = gpt(decoder_input_batch, target_mask=mask)
    
    print(output, output.shape)
