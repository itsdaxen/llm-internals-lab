import torch

from llm_internals.data_loading.gpt_dataset import create_dataloader
from llm_internals.embeddings.embeddings import build_batch_input_embeddings
from llm_internals.attention.attention import simple_compute_attention


def test_simple_compute_attention_preserves_embedding_shape():
    vocab_size = 50257
    embedding_dim = 4
    max_length = 4

    raw_text = (
        "This is a sample text, and is meant to test if our process "
        "use dataloader to retrive tokens and then create batch embeddings and then calculate attention"
    )

    dataloader = create_dataloader(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    pos_embedding_layer = torch.nn.Embedding(max_length, embedding_dim)

    for inputs, _ in dataloader:
        input_embeddings = build_batch_input_embeddings(
            inputs,
            token_embedding_layer,
            pos_embedding_layer,
        )
        context_vectors = simple_compute_attention(input_embeddings)
        assert context_vectors.shape == input_embeddings.shape
