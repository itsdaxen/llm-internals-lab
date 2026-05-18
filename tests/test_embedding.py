import torch

from llm_internals.embeddings.embeddings import build_batch_input_embeddings
from llm_internals.data_loading.gpt_dataset import create_dataloader


def test_build_batch_input_embeddings_returns_expected_shape():
    vocab_size = 50257
    embedding_dim = 4
    max_length = 4

    raw_text = (
        "This is a sample text, and is meant to test if our process "
        "use dataloader to retrive tokens and then create batch embeddings"
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

        assert input_embeddings.shape == torch.Size([1, 4, 4])
