# LLM Systems Lab

From-scratch implementations of GPT-style LLM components, focused on building, testing, and documenting the core systems pipeline behind transformer language models.

This repository is not intended to be a production-scale LLM. It is a systems lab for implementing transformer internals, validating tensor shape flows, testing components, and later profiling inference behavior, memory usage, latency, and deployment tradeoffs.

The long-term direction is to connect model internals with practical AI systems concerns: inference, evaluation, observability, deployment constraints, and edge/cloud execution tradeoffs.

## Status

This repository is in active development.

Currently implemented:

- Regex-based tokenization
- Vocabulary building
- Special token handling
- GPT-style sliding-window dataset and dataloader
- Batch token + positional input embeddings
- Simplified self-attention
- Unit tests for tokenizer, data sampling, embeddings, and attention

Current test status:

```bash
pytest
# 8 passed