# System Architecture — Offline Hindi ↔ Mundari Neural Machine Translation Engine

## 1. Overview
The Offline Hindi ↔ Mundari Neural Machine Translation (NMT) system provides bidirectional text translation between Hindi and Mundari without cloud network dependencies. Designed for low-resource edge deployment, the system integrates clean dataset preprocessing, shared subword BPE tokenization, 3-layer Transformer NMT models, dynamic INT8 ONNX quantization, and an offline Android mobile runtime.

---

## 2. Component Pipeline Diagram

```
                             [ User Text Input ]
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    Directional Selection      │
                      │    (Hindi ↔ Mundari)          │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    SentencePiece Tokenizer    │
                      │    (Shared 8K BPE Vocab)      │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │   Quantized ONNX Engine       │
                      │   (Dynamic INT8 Runtime)      │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │     Translated Output         │
                      └───────────────────────────────┘
```

---

## 3. Core Subsystems

### 3.1 Data Subsystem
- **Parallel Datasets:** AdiBhashaa (20,000 pairs) + Karya (10,268 pairs)
- **Combined Corpus:** 30,225 unique pairs after exact duplicate filtering
- **Split Ratio:** 80% Train (24,180), 10% Dev (3,022), 10% Test (3,023)
- **Script:** Devanagari representation for both Hindi and Mundari

### 3.2 Tokenization Subsystem
- **Algorithm:** SentencePiece Byte-Pair Encoding (BPE)
- **Vocabulary Size:** 8,000
- **Character Coverage:** 1.0 (Zero OOV character loss)
- **Special Tokens:** `<unk>=0`, `<s>=1`, `</s>=2`, `<pad>=3`

### 3.3 Translation Engine (PyTorch / ONNX)
- **Architecture:** Sequence-to-Sequence Transformer with causal masking
- **Embedding Dimension ($d_{model}$):** 256
- **Attention Heads ($n_{head}$):** 4
- **Encoder / Decoder Layers:** 3 layers / 3 layers
- **Feedforward Dimension ($d_{ff}$):** 1024
- **Dropout:** 0.1

### 3.4 Edge Runtime & Optimization Subsystem
- **Model Representation:** ONNX (Open Neural Network Exchange) format
- **Quantization:** Dynamic INT8 weight quantization (QUInt8)
- **Compression Efficiency:** 140.48 MB (PyTorch FP32) → 45.06 MB (ONNX FP32) → 13.62 MB (ONNX INT8)
- **Target Platform:** Offline Android OS via ONNX Runtime Mobile SDK
