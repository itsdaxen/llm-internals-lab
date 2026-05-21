import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Scaled dot-product self-attention using trainable Q/K/V linear projections.

    This module projects input token embeddings into query, key, and value
    representations using PyTorch Linear layers. It then computes attention scores
    from query-key similarity, normalizes them into attention weights, and uses
    those weights to produce context vectors from the values.

    Args:
        d_in: Size of each input token embedding.
        d_out: Size of each query, key, value, and context vector.
        qkv_bias: Whether to include bias terms in the query, key, and value projections.

    Input:
        x: Tensor of shape [batch_size, max_length, d_in].

    Returns:
        Tensor of shape [batch_size, max_length, d_out] containing context vectors.
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # Create trainable linear projection matrix for creating query vectors
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        # reate trainable linear projection matrix for creating key vectors
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        # reate trainable linear projection matrix for creating value vectors
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """
        Compute context-aware token representations using self-attention.

        Args:
            x: Input tensor of shape [batch_size, max_length, d_in].

        Returns:
            Context vectors of shape [batch_size, max_length, d_out].
        """

        # Project each input token into key, query, and value spaces.
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Compute raw attention scores by comparing each query with every key.
        attn_scores = queries @ keys.transpose(-2, -1)

        # Scale scores to prevent softmax from becoming too sharp, then normalize across the token dimension.
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Mix value vectors according to the attention weights and create a weighted combination of all value vectors
        context_vec = attn_weights @ values

        return context_vec
