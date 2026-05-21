# Model Release Guide

This project can be published on GitHub while keeping trained model artifacts outside
the Git repository. The recommended flow is:

1. Push source code to GitHub.
2. Package trained models as archives.
3. Upload those archives to a GitHub Release.
4. Keep model paths configurable through `.env`.

## Important License Notice

All model artifacts, datasets, configuration files, adapters, weights, and source
code associated with this project are provided under an "All rights reserved"
license.

Downloading release assets does not grant permission to copy, redistribute,
modify, sublicense, publish, use commercially, or create derivative works from
the software or model artifacts.

Any use of this project or its trained models requires prior written permission
from the author:

```text
KonstantinBab
```

## Recommended Release Assets

Prepare separate archives so users can download only what they need:

```text
graphit-embedder-berta.zip
graphit-selector-lora.zip
graphit-classifier.zip
graphit-vikr-reference.zip
```

Suggested local layout after download:

```text
GraphIT/
├── models/
│   ├── berta_finetuned_v6_second/final/
│   ├── selector_finetuned_7b_v6/final/
│   └── classifier_llm_v7/final/
└── docs/
    └── vikr_full.xlsx
```

Then set `.env`:

```text
GRAPHIT_EMBEDDER_PATH=models/berta_finetuned_v6_second/final
GRAPHIT_SELECTOR_PATH=models/selector_finetuned_7b_v6/final
GRAPHIT_CLASSIFIER_PATH=models/classifier_llm_v7/final
GRAPHIT_CODES_FILE=docs/vikr_full.xlsx
```

## Create Archives

PowerShell examples:

```powershell
Compress-Archive -Path "G:\Gratio_No_Gratio\augmentation\berta_finetuned_v6_second\final\*" `
  -DestinationPath "graphit-embedder-berta.zip" -Force

Compress-Archive -Path "G:\Gratio_No_Gratio\train_selector\selector_finetuned_7b_v6\final\*" `
  -DestinationPath "graphit-selector-lora.zip" -Force

Compress-Archive -Path "G:\Gratio_No_Gratio\train_classification\classifier_llm_v7\final\*" `
  -DestinationPath "graphit-classifier.zip" -Force

Compress-Archive -Path "G:\Gratio_No_Gratio\gradio\docs\vikr_full.xlsx" `
  -DestinationPath "graphit-vikr-reference.zip" -Force
```

## Upload To GitHub Release

After the repository is pushed, install GitHub CLI and log in:

```powershell
gh auth login
```

Create a release and upload all model archives:

```powershell
gh release create v0.1.0 `
  graphit-embedder-berta.zip `
  graphit-selector-lora.zip `
  graphit-classifier.zip `
  graphit-vikr-reference.zip `
  --repo KonstantinBab/GraphIT `
  --title "GraphIT v0.1.0 - trained model artifacts" `
  --notes-file RELEASE_NOTES.md
```

If the release already exists:

```powershell
gh release upload v0.1.0 `
  graphit-embedder-berta.zip `
  graphit-selector-lora.zip `
  graphit-classifier.zip `
  graphit-vikr-reference.zip `
  --repo KonstantinBab/GraphIT
```

## Practical Note

For a public GitHub repository, release assets can be downloaded by anyone who
can access the repository. The license restricts legal use and redistribution,
but it is not a technical access-control mechanism.

If you need technical control over downloads, use a private repository, private
GitHub Release, private Hugging Face repository, cloud storage with expiring
links, or manual access approval.
