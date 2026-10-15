# Multilingual Disinformation Detection: A Diagnostic Framework

Diagnostic framework for detecting class-imbalance failure in multilingual and domain-specific misinformation detection.

## Research Question

Can we detect silent failures in misinformation detection models where high accuracy masks minority-class collapse?

Our diagnostic: The accuracy-macro-F1 gap.

## Key Finding

| Dataset | Balance Ratio | Accuracy | Macro-F1 | Gap | Verdict |
|---------|---------------|----------|----------|-----|---------|
| X-Fact Hindi (binary) | 5.88 | 0.877 | 0.465 | 0.412 | Failure |
| X-Fact Spanish (binary) | 1.x | 0.744 | 0.672 | 0.072 | Healthy |
| X-Fact Portuguese (binary) | ~1 | 0.664 | 0.664 | 0.000 | Healthy |
| US Immigrant (RoBERTa) | 1.25 | 0.953 | 0.952 | 0.0007 | Perfect |

Rule: Gap > 0.30 means check for class imbalance.

## Structure

    notebooks/                    Training scripts
    results/                      Experiment results
    paper/                        Paper draft

## Models on HuggingFace

- XLM-R Binary: https://huggingface.co/nusaiba19/xlmr-xfact-binary
- XLM-R 7-class: https://huggingface.co/nusaiba19/xlmr-xfact-7class
- RoBERTa Immigrant: https://huggingface.co/nusaiba19/roberta-immigrant-hate-speech

## Training Setup

### XLM-R (X-Fact)
- Model: xlm-roberta-base
- LR: 2e-5, Batch: 16, Max len: 128, Epochs: 5
- Device: Google Colab T4

### RoBERTa (US Immigrant)
- Model: roberta-base
- LR: 2e-5, Batch: 16, Max len: 256, Epochs: 3
- Class weights: balanced
- Device: Apple M4 (MPS backend)
