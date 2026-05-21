# GraphIT v0.1.0 - trained model artifacts

This release contains optional trained artifacts for running the full GraphIT
pipeline locally.

GraphIT v0.1.0 is a working MVP. The training data covered RD disciplines
АС, КЖ, КМ, with ТХ represented in a small volume. Quality should be validated
separately for other disciplines and document formats.

## Assets

- `graphit-embedder-berta.zip` - BERTA embedder for semantic search.
- `graphit-selector-lora.zip` - LLM Selector LoRA adapter.
- `graphit-classifier.zip` - work/material classifier adapter.
- `graphit-vikr-reference.zip` - ВиКР reference file.

## License Notice

All rights reserved.

Downloading these files does not grant permission to copy, redistribute, modify,
sublicense, publish, use commercially, or create derivative works from the model
artifacts or source code.

Use is allowed only with prior written permission from the author:

```text
KonstantinBab
```

## Setup

Unpack the archives into the project folder and configure `.env`:

```text
GRAPHIT_EMBEDDER_PATH=models/berta_finetuned_v6_second/final
GRAPHIT_SELECTOR_PATH=models/selector_finetuned_7b_v6/final
GRAPHIT_CLASSIFIER_PATH=models/classifier_llm_v7/final
GRAPHIT_CODES_FILE=docs/vikr_full.xlsx
```
