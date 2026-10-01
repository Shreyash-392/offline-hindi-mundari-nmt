# Technical Experiment Log & Ablation Records

## Overview
This document records every ML experiment, architecture variant, scheduler change, and ablation study performed on the Offline Hindi ↔ Mundari NMT system.

---

## Experiment A: Unmasked Transformer Training
- **Objective:** Train initial baseline Transformer without causal target masks (`tgt_mask`).
- **Result:** Severe EOS collapse and repetitive looping (e.g. repeating single tokens up to 80 positions).
- **Decision:** REJECTED. Causal masking (`torch.triu`) is strictly mandatory.

---

## Experiment B: Corrected Causal & Padding Masked Transformer
- **Objective:** Introduce explicit target causal masks (`tgt_mask`) and key padding masks (`src_key_padding_mask`, `tgt_key_padding_mask`).
- **Result:** Fully eliminated EOS collapse and infinite loops.
- **Decision:** ACCEPTED as core architectural requirement.

---

## Experiment C: Official Baseline Training (10 Epochs, AdamW)
- **Architecture:** $d_{model}=256, n_{head}=4, N_{enc}=3, N_{dec}=3, d_{ff}=1024, \text{dropout}=0.1$
- **Optimizer:** AdamW ($lr=0.0003, \text{weight\_decay}=0.01$)
- **Best Validation Loss:** `5.1180` (Epoch 9)
- **Test Metrics (Hindi → Mundari):** Greedy BLEU `0.6303`, Beam=3 BLEU `0.8264`, chrF++ `15.5655`, Repetition `17.86%`
- **Decision:** ACCEPTED as official baseline checkpoint (`checkpoints/transformer_best.pt`).

---

## Experiment D: ReduceLROnPlateau Learning Rate Scheduler
- **Variable Changed:** Added `torch.optim.lr_scheduler.ReduceLROnPlateau` (factor 0.5, patience 2).
- **Best Validation Loss:** `5.1328` (Worse than baseline 5.1180).
- **Decision:** REJECTED.

---

## Experiment E: Large Transformer Architecture Expansion
- **Variable Changed:** $d_{model}=384, n_{head}=6, N_{enc}=4, N_{dec}=4, d_{ff}=1536$ (~25.8M parameters).
- **Best Validation Loss:** `5.7472` (Significantly worse due to low-resource over-fitting).
- **Decision:** REJECTED.

---

## Experiment F: Bidirectional Mundari → Hindi NMT Model
- **Objective:** Train independent reverse translation model (Mundari → Hindi).
- **Architecture:** $d_{model}=256, n_{head}=4, N_{enc}=3, N_{dec}=3, d_{ff}=1024$
- **Best Validation Loss:** `4.5396` (Epoch 9)
- **Test Metrics (Mundari → Hindi):** Greedy BLEU `2.0668`, Beam=3 BLEU `1.9742`, chrF++ `15.5807`, Repetition `6.33%`
- **Decision:** ACCEPTED as official reverse checkpoint (`checkpoints/transformer_mundari_hindi_best.pt`).

---

## Experiment G: Dictionary-Augmented Training Set (Pronoun & Core Lexicon)
- **Objective:** Inject 56 core lexicon dictionary pairs (including `"तुम्हारा"` $\rightarrow$ `"अमअः"`) into training dataset.
- **Best Validation Loss:** `5.1088` (Epoch 8)
- **Single-Word Accuracy:** 100% correct single-word lexicon lookup (e.g. `"तुम्हारा"` $\rightarrow$ `"अमअः"`) while retaining full-sentence translation capacity.
- **Decision:** ACCEPTED (`checkpoints/transformer_augmented_best.pt`).
