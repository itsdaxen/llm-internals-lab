import torch


def simple_compute_attention(input_tensors: torch.Tensor) -> torch.Tensor:
    """
    Compute simplified self-attention context vectors.

    Args:
        input_tensors: Tensor of shape [batch_size, max_length, embedding_dim].

    Returns:
        Context vectors of shape [batch_size, max_length, embedding_dim].
    """

    # Compute raw attention scores by comparing every token with every other token.
    attn_scores = input_tensors @ input_tensors.transpose(-2, -1)

    # Normalize each row into attention weights.
    attn_weights = torch.softmax(attn_scores, dim=-1)

    # Compute one context vector per token as a weighted sum of all input vectors.
    context_vectors = attn_weights @ input_tensors

    return context_vectors
