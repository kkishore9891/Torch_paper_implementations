import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils import InputEmbedding, PositionalEncoding


class MultiHeadAttention(nn.Module):
    """Generic Multi-Head Self-Attention module.

    Splits the query, key, and value tensors into multiple heads and computes
    scaled dot-product attention. An optional mask can be applied to the scores.
    """
    def __init__(self, n_heads: int = 5, embedding_dim: int = 10,
                 attn_dropout: float = 0.1, res_dropout: float = 0.1):
        """
        Args:
            n_heads: Number of attention heads.
            embedding_dim: Dimensionality of the input embeddings.
            attn_dropout: Dropout probability applied to attention scores.
            res_dropout: Dropout probability applied to the attention output.
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = embedding_dim

        if embedding_dim % n_heads != 0:
            raise ValueError(
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

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            Q: Query tensor of shape (batch_size, seq_len, embedding_dim).
            K: Key tensor of shape (batch_size, seq_len, embedding_dim).
            V: Value tensor of shape (batch_size, seq_len, embedding_dim).
            mask: Optional boolean mask of shape (seq_len, seq_len).

        Returns:
            Tensor: Output of the multi-head attention layer.
        """
        batch_size, query_len, _ = Q.shape
        key_len = K.shape[1]

        # Linear projections and reshape for multiple heads.
        query = self.q_proj(Q).reshape(batch_size, query_len, self.n_heads, self.d_q)
        query = query.permute(0, 2, 1, 3)

        key = self.k_proj(K).reshape(batch_size, key_len, self.n_heads, self.d_k)
        key = key.permute(0, 2, 1, 3)

        value = self.v_proj(V).reshape(batch_size, key_len, self.n_heads, self.d_v)
        value = value.permute(0, 2, 1, 3)

        # Scaled dot-product attention.
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            # Mask is assumed to be broadcastable to (query_len, key_len)
            scores = scores.masked_fill(mask[:query_len, :query_len], float('-inf'))
        attn = self.attn_dropout(self.softmax(scores))

        # Combine attention output.
        context = torch.matmul(attn, value)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, query_len, self.d_model)
        output = self.res_dropout(self.linear_out(context))
        return output


class DecoderBlock(nn.Module):
    """Single Transformer decoder block with self-attention, residual connections,
    layer normalization, and a feed-forward network.
    """
    def __init__(self, n_heads: int = 5, embedding_dim: int = 10,
                 ff_multiplier: int = 4, dropout: float = 0.1):
        """
        Args:
            n_heads: Number of attention heads.
            embedding_dim: Dimensionality of the token embeddings.
            ff_multiplier: Factor to determine the size of the hidden layer in the feed-forward network.
            dropout: Dropout probability.
        """
        super(DecoderBlock, self).__init__()
        self.mhsa = MultiHeadAttention(n_heads=n_heads, embedding_dim=embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_multiplier * embedding_dim),
            nn.GELU(),
            nn.Linear(ff_multiplier * embedding_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim).
            mask: Optional mask for the attention layers.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim).
        """
        # Self-attention with residual connection.
        attn = self.mhsa(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + attn

        # Feed-forward network with residual connection.
        ff = self.dropout(self.feed_forward(self.norm2(x)))
        x = x + ff
        return x


class DecoderStack(nn.Module):
    """Stacks multiple decoder blocks with input embeddings and positional encodings."""
    def __init__(self, src_vocab_len: int = 10, max_seq_len: int = 5,
                 n_decoders: int = 6, n_heads: int = 5, embedding_dim: int = 10,
                 ff_multiplier: int = 4, dropout: float = 0.1):
        """
        Args:
            src_vocab_len: Vocabulary length of the source data.
            max_seq_len: Maximum sequence length.
            n_decoders: Number of decoder blocks to stack.
            n_heads: Number of attention heads per block.
            embedding_dim: Dimensionality of the token embeddings.
            ff_multiplier: Factor for the hidden layer size in the feed-forward network.
            dropout: Dropout probability.
        """
        super(DecoderStack, self).__init__()
        self.decoder_input_embedding = InputEmbedding(vocab_len=src_vocab_len,
                                                      embedding_dim=embedding_dim)
        self.decoder_position_embeddings = PositionalEncoding(max_seq_len=max_seq_len,
                                                              embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_list = nn.ModuleList([
            DecoderBlock(n_heads=n_heads, embedding_dim=embedding_dim,
                         ff_multiplier=ff_multiplier, dropout=dropout)
            for _ in range(n_decoders)
        ])

    def forward(self, decoder_input_batch: torch.Tensor,
                target_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            decoder_input_batch: Tensor of shape (batch_size, seq_len) with token indices.
            target_mask: Optional mask to apply in attention layers.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim).
        """
        x = self.decoder_input_embedding(decoder_input_batch)
        # Add positional encoding and apply dropout.
        x = x + self.decoder_position_embeddings[:x.shape[1]]
        x = self.dropout(x)

        for block in self.layer_list:
            x = block(x, mask=target_mask)
        return x


class GPT(nn.Module):
    """Transformer-based language model using a stacked decoder architecture."""
    def __init__(self, src_vocab_len: int = 10, targ_vocab_len: int = 10,
                 max_seq_len: int = 5, dropout: float = 0.1, n_decoders: int = 6,
                 n_heads: int = 5, embedding_dim: int = 10, ff_multiplier: int = 4):
        """
        Args:
            src_vocab_len: Vocabulary length of the source data.
            targ_vocab_len: Vocabulary length of the target data.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.
            n_decoders: Number of decoder blocks to stack.
            n_heads: Number of attention heads per block.
            embedding_dim: Dimensionality of the token embeddings.
            ff_multiplier: Factor for the hidden layer size in the feed-forward network.
        """
        super(GPT, self).__init__()
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
    def generate(self, x: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 do_sample: bool = False, top_k: int = None) -> torch.Tensor:
        """
        Generates a sequence of tokens given a conditioning input.

        Args:
            x: Conditioning sequence (LongTensor of shape (batch_size, t)).
            max_new_tokens: Number of tokens to generate.
            temperature: Temperature for scaling logits.
            do_sample: If True, sample from the probability distribution; otherwise, use greedy decoding.
            top_k: If provided, restrict sampling to the top k tokens.

        Returns:
            Tensor: The extended sequence including generated tokens.
        """

        # Inspired from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        for _ in range(max_new_tokens):
            # Ensure sequence stays within block size.
            if x.size(1) > self.block_size:
                x = x[:, -self.block_size:]

            logits = self(x)
            # Scale the logits for the last token.
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_values, _ = torch.topk(logits, top_k)
                logits[logits < top_values[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                x_next = torch.multinomial(probs, num_samples=1)
            else:
                _, x_next = torch.topk(probs, k=1, dim=-1)

            x = torch.cat((x, x_next), dim=1)
        return x

    def forward(self, decoder_input_batch: torch.Tensor,
                target_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the GPT model.

        Args:
            decoder_input_batch: Tensor of shape (batch_size, seq_len) with token indices.
            target_mask: Optional mask for attention layers.

        Returns:
            Tensor: Logits of shape (batch_size, seq_len, targ_vocab_len).
        """
        batch_size, seq_len = decoder_input_batch.size()
        if seq_len > self.block_size:
            raise ValueError("Input exceeds context length!")
        decoder_output = self.decoder_stack(decoder_input_batch, target_mask)
        logits = self.prediction_layer(self.norm(decoder_output))
        return logits


if __name__ == "__main__":
    # Configuration.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab_len = 10
    targ_vocab_len = 10
    max_seq_len = 1000
    dropout = 0.1
    n_decoders = 12
    n_heads = 12
    embedding_dim = 768
    ff_multiplier = 4

    # Sample input (batch_size, seq_len).
    decoder_input_batch = torch.tensor(
        [[6, 1, 2, 3, 4],
         [5, 6, 7, 8, 9],
         [9, 8, 7, 6, 5],
         [4, 3, 2, 1, 0]], dtype=torch.long, device=device)

    # Create an upper-triangular mask for causal self-attention.
    mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1).to(device)

    # Instantiate the model and move it to the device.
    gpt = GPT(
        src_vocab_len=src_vocab_len,
        targ_vocab_len=targ_vocab_len,
        max_seq_len=max_seq_len,
        dropout=dropout,
        n_decoders=n_decoders,
        n_heads=n_heads,
        embedding_dim=embedding_dim,
        ff_multiplier=ff_multiplier
    )
    gpt.to(device)
    summary(gpt)

    # Forward pass.
    output = gpt(decoder_input_batch, target_mask=mask)
    print(output, output.shape)
