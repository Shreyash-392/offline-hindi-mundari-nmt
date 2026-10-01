# Patent-Oriented Technical Record — Offline Low-Resource NMT

> [!NOTE]
> This document preserves detailed technical engineering implementation evidence, data flow structures, architecture decisions, and experimental benchmarks to assist future prior-art and technical patentability assessments. Do NOT make unsupported legal claims.

---

## 1. Technical Problem
Translating low-resource indigenous languages (such as Mundari) on edge mobile hardware presents severe technical barriers:
1. **Extreme Data Sparsity:** Standard models divergence on small parallel datasets (<35,000 pairs).
2. **Agglutinative Morphology Subword Explosion:** Rare word OOV errors when using standard tokenization.
3. **Edge Memory & Latency Constraints:** Unquantized neural translation engines consume >140 MB RAM and exceed thermal/latency limits on mobile CPUs without cloud connectivity.

---

## 2. Technical System Solution & Pipeline Combination

```
Low-Resource Parallel Data (AdiBhashaa + Karya)
       ↓
Data Cleaning & De-duplication Protocol (30,225 pairs)
       ↓
Shared Devanagari BPE Subword Tokenization (8,000 Vocab)
       ↓
Capacity-Calibrated Transformer Architecture (256-dim, 3-layer)
       ↓
Explicit Causal & Key-Padding Masking
       ↓
Bidirectional Independent Dual-Checkpoint Models
       ↓
Dictionary-Augmented Fine-Tuning Pipeline
       ↓
TorchScript Graph Tracing & ONNX Export (opset 14)
       ↓
Dynamic INT8 Model Quantization (81.1% Compression)
       ↓
Offline Android ONNX Mobile Engine (0 Network Sockets)
```

---

## 3. Potential Differentiating Technical Features for Future Patent Review

1. **Subword BPE Morphology Matching for Austroasiatic Script Alignment:**
   A joint SentencePiece subword representation achieving 1.0 character coverage over shared Devanagari representation for Mundari and Hindi.

2. **Capacity-Calibrated Compact Transformer Topology:**
   A specialized 3-layer, 4-head, 256-embedding dimension Transformer NMT topology tuned to avoid over-fitting and validation divergence under a strict <31,000 parallel pair constraint.

3. **Hybrid Dictionary-Augmented NMT Retraining Protocol:**
   Injecting repeated single-word lexicon pairs directly into the parallel sentence training stream to resolve isolated pronoun/vocabulary translation failures in seq2seq attention networks without degrading full-sentence BLEU.

4. **Zero-Network Offline Mobile Machine Translation Runtime:**
   The integration of dynamic INT8 ONNX weight quantization with local subword BPE tokenization executing on Android mobile hardware with zero network permissions.

---

## 4. Empirical Performance Evidence

| Metric | Hindi → Mundari | Mundari → Hindi |
| :--- | :---: | :---: |
| **Best Validation Loss** | 5.1088 | **4.5396** |
| **Test BLEU (Beam=3)** | 0.8264 | **2.1333** |
| **Test BLEU (Beam=5)** | 0.8264 | **2.1470** |
| **Test chrF++ Score (Beam=3)** | 15.5655 | **15.9634** |
| **Test chrF++ Score (Beam=5)** | 15.5655 | **16.0008** |
| **Repetition % (Beam=3)** | 17.86% | **7.34%** |
| **Empty Output %** | 0.00% | **0.00%** |
| **PyTorch Checkpoint Size** | 140.48 MB | 140.48 MB |
| **ONNX INT8 Size** | 25.34 MB | 25.34 MB |
| **Footprint Reduction** | **81.1%** | **81.1%** |
| **CPU Latency per Sentence** | ~250–300 ms | ~240–270 ms |
| **Network Sockets** | **0 (100% Offline)** | **0 (100% Offline)** |
