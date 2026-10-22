# The Lineage of BERT

*Every fact in this note was verified by live lookup (Hugging Face Hub, arXiv abstract pages, papers' own LaTeX sources, GitHub API) on 2026-10-22.*

*Note: this file is stored in a repo because the Gist API is not accessible with the current GitHub token (403 "Resource not accessible by personal access token"). It is intended to be the `bert-lineage.md` gist.*

---

## 1. The Anchor Model — `google-bert/bert-base-uncased`

**Hub entry:** https://hf.co/google-bert/bert-base-uncased

| Stat | Value (live) |
|---|---|
| Pipeline task | `fill-mask` |
| Downloads | 3,268.9M (~3.27 billion) |
| Likes | 3,043 |
| Parameters | 110.1M |
| Library | `transformers` |
| License | apache-2.0 |
| Last updated | 19 Feb 2024 |
| Model class | `AutoModelForMaskedLM` |

### What the repository actually ships

**Weight formats:**
- `model.safetensors` — 440.4 MB (safetensors)
- `pytorch_model.bin` — 440.5 MB (PyTorch)
- `tf_model.h5` — 536.1 MB (TensorFlow)
- `flax_model.msgpack` — 438.1 MB (Flax/JAX)
- `model.onnx` — 532.1 MB (ONNX)
- `rust_model.ot` — 534.2 MB (Rust)
- `coreml/fill-mask/` — directory of Core ML weights

**Config and tokenizer files:**
- `config.json` (570 B)
- `tokenizer.json` (466.1 KB)
- `tokenizer_config.json` (48 B)
- `vocab.txt` (231.5 KB)

Plus `LICENSE` (Apache-2.0) and `README.md`. Tags include `bert`, `fill-mask`, `en`, `dataset:bookcorpus`, `dataset:wikipedia`, and **`arxiv:1810.04805`**.

---

## 2. The BERT Paper

### Confirmed identifier
The Hub entry itself carries the tag `arxiv:1810.04805`, tying the model to the original BERT paper.

- **arXiv ID:** 1810.04805 (cs.CL) — https://arxiv.org/abs/1810.04805
- **Official title (from the public arXiv abstract page):** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- **Cross-check vs. the paper's own LaTeX source:** the source contains `\title{\textbf{BERT}: Pre-training of Deep Bidirectional Transformers for \\ Language Understanding}` — identical. ✔
- **Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI Language)

### Section outline (as the paper prints it)

```
Abstract
1 Introduction
2 Related Work
  2.1 Unsupervised Feature-based Approaches
  2.2 Unsupervised Fine-tuning Approaches
  2.3 Transfer Learning from Supervised Data
3 BERT
  [Model Architecture / Input/Output Representations — unnumbered headings]
  3.1 Pre-training BERT
      [Task #1: Masked LM / Task #2: Next Sentence Prediction (NSP) — unnumbered headings]
  3.2 Fine-tuning BERT
4 Experiments
  4.1 GLUE
  4.2 SQuAD v1.1
  4.3 SQuAD v2.0
  4.4 SWAG
5 Ablation Studies
  5.1 Effect of Pre-training Tasks
  5.2 Effect of Model Size
  5.3 Feature-based Approach with BERT
6 Conclusion
References
Appendix A  Additional Details for BERT
  A.1 Illustration of the Pre-training Tasks
  A.2 Pre-training Procedure
  A.3 Fine-tuning Procedure
  A.4 Comparison of BERT, ELMo, and OpenAI GPT
  A.5 Illustrations of Fine-tuning on Different Tasks
Appendix B  Detailed Experimental Setup
  B.1 Detailed Descriptions for the GLUE Benchmark Experiments
Appendix C  Additional Ablation Studies
  C.1 Effect of Number of Training Steps
  C.2 Ablation for Different Masking Procedures
```

### Headline experiment numbers
New state-of-the-art results on **eleven NLP tasks**, including:
- **GLUE score: 80.5** (7.7 point absolute improvement)
- **MultiNLI accuracy: 86.7%** (4.6 point absolute improvement)
- **SQuAD v1.1 question answering Test F1: 93.2** (1.5 point absolute improvement)
- **SQuAD v2.0 Test F1: 83.1** (5.1 point absolute improvement)

---

## 3. The Progenitor — Vaswani et al. (2017)

### How it was tracked down
The BERT paper's Introduction notes that OpenAI GPT uses "a left-to-right architecture, where every token can only attend to previous tokens in the self-attention layers of the Transformer~\cite{vaswani-etal:2017:_atten}". The paper's reference list resolves that citation to **arXiv:1706.03762**.

- **arXiv ID:** 1706.03762 (cs.CL, cs.LG) — https://arxiv.org/abs/1706.03762
- **Official title (from the public arXiv abstract page):** *Attention Is All You Need*
- **Cross-check vs. the paper's own LaTeX source:** the source contains `\title{Attention Is All You Need}` — identical. ✔
- **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

### Section outline (as the paper prints it)

```
Abstract
1 Introduction
2 Background
3 Model Architecture
  3.1 Encoder and Decoder Stacks
  3.2 Attention
    3.2.1 Scaled Dot-Product Attention
    3.2.2 Multi-Head Attention
    3.2.3 Applications of Attention in our Model
  3.3 Position-wise Feed-Forward Networks
  3.4 Embeddings and Softmax
  3.5 Positional Encoding
4 Why Self-Attention
5 Training
  5.1 Training Data and Batching
  5.2 Hardware and Schedule
  5.3 Optimizer
  5.4 Regularization
6 Results
  6.1 Machine Translation
  6.2 Model Variations
  6.3 English Constituency Parsing
7 Conclusion
References
```

### Headline machine-translation results
- **WMT 2014 English→German: 28.4 BLEU** — the big Transformer model beats all previously published models and ensembles by more than 2.0 BLEU.
- **WMT 2014 English→French: 41.8 BLEU** — a new single-model state of the art, after only 3.5 days of training on 8 GPUs (a fraction of the training cost of the best prior models).
- The base model alone: 27.3 BLEU (EN-DE) and 38.1 BLEU (EN-FR) at a fraction of the training cost of prior competitive models.
- Also generalized successfully to English constituency parsing (both large and limited training data).

---

## 4. GitHub Actions Artifact Note

Checked all repositories under the GitHub account `HeziLiu` (HeziLiu/Little-rookie, HeziLiu/Big-Buzzard, HeziLiu/CodingInterviewChinese2, HeziLiu/awesome-chatgpt-prompts-zh) — **none have any Actions workflows set up.** The accessible workflows live in the `mcpmark-eval-liuhezi` organization.

**Newest downloadable workflow-run artifact found:**
- **Name:** `release-check-report`
- **Repo:** mcpmark-eval-liuhezi/release-dry-run-3 ("Throwaway repo for rehearsing the new release checklist")
- **Workflow / run:** "Release Checks" (`.github/workflows/release-checks.yml`), run #1, ID 39701037830, triggered manually (workflow_dispatch), conclusion: success
- **Created:** 2026-10-22T00:40:05Z (202 bytes, not expired)
- **What's inside:** a single file, `README.md` — the repository's own README, which reads: `# release-dry-run-3` / `Throwaway repo for rehearsing the new release checklist`. The workflow's final step simply uploads `README.md` as the `release-check-report` artifact.

(Runner-up, for reference: the `build-info` artifact in mcpmark-eval-liuhezi/hezi-pipeline-demo, created 2026-10-21T22:36:54Z — older than the one above.)

---

## 5. Lineage Summary

```
Vaswani et al. (2017)                 Devlin et al. (2018)                 google-bert/bert-base-uncased
"Attention Is All You Need"     ──►    "BERT: Pre-training of Deep       ──►    Hub model (fill-mask, 110.1M params,
arXiv:1706.03762                       Bidirectional Transformers for            3.27B downloads, 3,043 likes)
Transformer encoder                    Language Understanding"                   ships weights in 7 formats +
WMT14 EN-DE 28.4 BLEU                  arXiv:1810.04805                          tokenizer files
WMT14 EN-FR 41.8 BLEU                  GLUE 80.5 / MNLI 86.7 /
                                       SQuAD v1.1 F1 93.2 / v2.0 F1 83.1
```

BERT's Introduction credits the Transformer architecture to Vaswani et al. (2017) (arXiv:1706.03762), and the Hub model entry itself is tagged `arxiv:1810.04805` — the chain is fully confirmed end to end.
