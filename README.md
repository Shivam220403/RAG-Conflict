# RAG-Conflict
Evaluating reference-free uncertainty signals (token entropy, semantic entropy) in RAG systems under conflicting retrieved evidence. Dissertation project.

## Datasets
### Natural Datasets, Entry Count & Role

| # | Dataset | HuggingFace ID | Entries (N) | Role / Category |
|---|---------|---------------|-------------|-----------------|
| 1 | **WikiContradict** | `ibm-research/Wikipedia_contradict_benchmark` | **253** | `conflict` — Real Wikipedia contradictions; forms the **Conflict Set** |
| 2 | **NQ-Open** | `google-research-datasets/nq_open` | **500** (sampled from 3,610) | `control` — Standard clean-context QA; forms the **Control Set** |
| 3 | **LLMBar** | `princeton-nlp/LLMBar` (Adversarial) | **~200** (adversarial subset) | `audit` — Tests positional & verbosity bias in LLM judges |
| 4 | **RAGTruth** | `wandb/RAGTruth` | **~18,000** | `calibration` — Word-level RAG hallucination examples for judge calibration |

#### Dataset-to-Pipeline Role Mapping
- `conflict`     → Conflict Set input  (WikiContradict + Synthetic Augmentation → N=503)
- `control`      → Control Set input   (NQ-Open 250 sampled → N=503 after augmentation)
- `audit`        → Judge Bias Audit    (LLMBar Adversarial)
- `calibration`  → Few-shot Judge Prompts (RAGTruth)

#### Conflict Set Target (Post-Augmentation)
| Source          | Type      | N   |
|-----------------|-----------|-----|
| WikiContradict  | Natural   | 253 |
| NQ-Open derived | Synthetic | 250 |
| **Total**       |           | **503** |
