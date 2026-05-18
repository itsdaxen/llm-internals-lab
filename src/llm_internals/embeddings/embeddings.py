import torch


def build_batch_input_embeddings(
    inputs: torch.Tensor,
    token_embedding_layer: torch.nn.Embedding,
    pos_embedding_layer: torch.nn.Embedding,
) -> torch.Tensor:
    """
    Build token + positional embeddings for one batch.

    Args:
        inputs: Tensor of shape [batch_size, max_length].

    Returns:
        Tensor of shape [batch_size, max_length, embedding_dim].
    """

    batch_size, max_length = inputs.shape

    token_embeddings = token_embedding_layer(inputs)

    pos_embeddings = pos_embedding_layer(torch.arange(max_length, device=inputs.device))

    input_embeddings = token_embeddings + pos_embeddings

    return input_embeddings
