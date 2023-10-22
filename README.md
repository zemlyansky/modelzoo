# modelzoo.js

The repo is based on [gpt-tfjs](https://github.com/zemlyansky/gpt-tfjs) and its TypeScript [fork](https://github.com/lukemovement/gpt-tfjs/tree/typescript-refactor) by [lukemovement](https://github.com/lukemovement). A lot of models share common layers and preprocessing methods so it makes sense to continue the work in a single repo rather than copy-pasting the same transformer architecture over and over again or having to maintain multiple repos for each layer. Currently not a lot to see here.

🔴 Work in progress 🔴

## Current models

- [x] GPT
- [x] Vision Transformer
- [ ] CLIP - in progress

## Custom layers

- [x] Logging
- [x] Range
- [x] Positional embedding
- [x] Class embedding
- [x] Slice
- [x] GELU
- [x] QuickGELU
- [x] Multihead attention
- [x] MLP
- [x] Residual Attention block
- [x] Transformer

## Optimizers

- [x] AdamW