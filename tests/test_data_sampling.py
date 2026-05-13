from llm_internals.data_loading.gpt_dataset import create_dataloader
import torch


def test_dataloader_returns_shifted_input_and_target_batches():
    raw_text = (
        "This is a sample text, and is meant to test if our process "
        "can tokenize this text properly and then make a dataset and return that data!"
    )

    dataloader = create_dataloader(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)

    inputs, targets = next(data_iter)

    assert inputs.shape == torch.Size([1, 4])
    assert targets.shape == torch.Size([1, 4])

    assert torch.equal(inputs, torch.tensor([[1212, 318, 257, 6291]]))
    assert torch.equal(targets, torch.tensor([[318, 257, 6291, 2420]]))
