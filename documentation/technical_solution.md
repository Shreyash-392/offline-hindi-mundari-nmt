# Technical Solution & Engineering Innovations

## 1. System Engineering Strategy

To resolve data scarcity, model over-fitting, and offline edge constraints, the project engineered a complete end-to-end NMT pipeline:

### 1.1 Shared Devanagari BPE Subword Tokenization
- Trained a joint SentencePiece BPE tokenizer with vocabulary size of 8,000 on train split only.
- Achieved **100% character coverage**, ensuring morphological subwords are shared effectively between Hindi and Mundari in Devanagari script.

### 1.2 Capacity-Calibrated 3-Layer Transformer Architecture
- Identified that standard Transformer-Large models (25.8M parameters, $d_{model}=384$, 4 layers) diverged (val loss 5.7472).
- Designed a capacity-calibrated **Compact Transformer** (11.7M parameters):
  - $d_{model} = 256$, $n_{head} = 4$
  - Encoder Layers = 3, Decoder Layers = 3
  - Feedforward Dim = 1024
- Achieved lowest validation loss (**4.5396** for Mundari → Hindi, **5.1180** for Hindi → Mundari).

### 1.3 Strict Causal & Padding Masking Protocol
- Implemented explicit combined target causal triangular masks (`torch.triu`) and key padding masks (`src_key_padding_mask`, `tgt_key_padding_mask`).
- Fully eliminated EOS collapse and repetitive looping observed in unmasked baseline experiments.

### 1.4 Dynamic INT8 ONNX Edge Quantization
- Exported PyTorch graph to ONNX operator graph using TorchScript tracing.
- Applied dynamic INT8 quantization to model weights:
  - FP32 Checkpoint: **140.48 MB**
  - INT8 ONNX Model: **13.62 MB** (**90.3% footprint reduction**)
- Achieved **100% exact output token match** between FP32 PyTorch and INT8 ONNX.
