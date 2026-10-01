# 🌐 Offline Hindi ↔ Mundari Neural Machine Translation (NMT) System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-brightgreen.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.15+-blueviolet.svg)](https://onnxruntime.ai/)
[![Android Compatible](https://img.shields.io/badge/Android-Offline_Capable-success.svg)](android/)

An end-to-end, resource-calibrated **Neural Machine Translation system for Hindi (Devanagari) ↔ Mundari (Mundari-Devanagari / Indigenous)**. Engineered specifically for edge deployment, low latency, zero network requirements, and high fidelity on low-resource parallel datasets.

---

## 🚀 Key Highlights

* **Bidirectional Translation:** Separate dedicated PyTorch & ONNX models for **Hindi → Mundari** and **Mundari → Hindi**.
* **Lexicon-Augmented Training:** Integrated core Mundari dictionary pairs to eliminate pronoun and isolated single-word translation errors while maintaining sentence-level BLEU scores.
* **INT8 Dynamic Quantization:** Model binary size reduced by **81.1%** (from 134.4 MB to **25.34 MB**) with zero precision loss.
* **Offline Android Mobile Engine:** Complete Kotlin Android application codebase leveraging `ONNX Runtime Mobile SDK (1.17.0)` with **0 network permission requirements**.
* **Low Latency:** Average CPU inference latency of **~260 ms** per sentence on low-power mobile/desktop ARM/x86 processors.

---

## 📐 Architecture & Model Calibration

| Parameter | Specification | Rationale / Optimization |
| :--- | :--- | :--- |
| **Model Type** | Sequence-to-Sequence Transformer | Custom PyTorch implementation |
| **Embedding Dimension ($d_{model}$)** | 256 | Prevents overfitting on ~30k sentences |
| **Attention Heads ($n_{head}$)** | 4 | 64-dim head capacity |
| **Encoder / Decoder Layers** | 3 / 3 | Optimal capacity for low-resource pairs |
| **Feed-Forward Dim ($d_{ff}$)** | 1024 | Compact non-linear expansion |
| **Subword Vocabulary** | 8,000 (BPE) | Shared SentencePiece tokenizer |
| **ONNX Format** | Opset 14 (INT8 Dynamic Quantized) | Optimized for ONNX Runtime Mobile |

---

## 📊 Benchmark & Evaluation Results

Evaluated on the official frozen **3,023 parallel sentence test set**:

| Direction | Decoding Method | BLEU Score | chrF++ Score | Memory Footprint | Avg Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hindi → Mundari** | Beam Search ($B=3$) | **0.8264** | **15.5655** | 25.34 MB (INT8) | ~258 ms |
| **Mundari → Hindi** | Beam Search ($B=5$) | **2.1470** | **16.0008** | 25.34 MB (INT8) | ~265 ms |

---

## 📁 Repository Structure

```
.
├── android/                   # Offline Kotlin Android Studio project codebase
│   └── app/src/main/java/com/nmt/offline/translation/
│       ├── MainActivity.kt    # UI controller & bidirectional toggle
│       ├── Tokenizer.kt       # Native SentencePiece subword wrapper
│       └── TranslationEngine.kt # ONNX Runtime Mobile inference handler
├── checkpoints/               # Trained weights & INT8 ONNX models
│   ├── transformer_hindi_mundari_int8.onnx   (25.34 MB)
│   └── transformer_mundari_hindi_int8.onnx   (25.34 MB)
├── data/                      # Dataset files (train, dev, frozen test)
│   ├── processed/             # Cleaned parallel corpus (25,300 pairs)
│   └── test/mundari/          # Frozen benchmark test set (3,023 pairs)
├── documentation/             # System documentation & patent technical records
│   ├── system_architecture.md
│   ├── optimization_results.md
│   ├── android_deployment.md
│   └── patent_technical_notes.md
├── evaluation/                # Test suite & benchmark evaluation scripts
│   ├── translate.py           # Hindi -> Mundari inference CLI
│   ├── translate_mundari_hindi.py # Mundari -> Hindi inference CLI
│   └── evaluate_beam.py       # BLEU & chrF++ benchmark runner
├── model/                     # PyTorch architecture & ONNX export pipelines
│   ├── transformer.py         # PyTorch Transformer model definition
│   ├── export_onnx.py         # TorchScript/ONNX export pipeline
│   └── quantize_onnx.py       # INT8 dynamic quantization script
├── preprocessing/             # Data cleaning & dictionary augmentation
├── tokenizer/                 # SentencePiece BPE tokenizer model (8,000 vocab)
├── training/                  # PyTorch training loops & dataset loaders
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # Project overview
```

---

## 🛠️ Quickstart & Setup

### 1. Environment Installation
```bash
git clone https://github.com/Shreyash-392/offline-hindi-mundari-nmt.git
cd offline-hindi-mundari-nmt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Command-Line Inference

#### Hindi → Mundari Translation:
```bash
python evaluation/translate.py --input "मैं अस्पताल जा रहा हूँ।"
```

#### Mundari → Hindi Translation:
```bash
python evaluation/translate_mundari_hindi.py --input "अइङ अस्पताल सेनोःतनाञ।"
```

### 3. Run Benchmark Evaluation
```bash
python evaluation/evaluate_beam.py --beam_size 3
```

### 4. Export & Quantize ONNX Models
```bash
# Export PyTorch weights to ONNX format
python model/export_onnx.py

# Perform INT8 dynamic quantization
python model/quantize_onnx.py
```

---

## 📱 Offline Android Application Setup

1. Open the `android/` directory in **Android Studio (Giraffe or newer)**.
2. Copy `tokenizer/hindi_mundari.model` and ONNX files from `checkpoints/` into `android/app/src/main/assets/`.
3. Build and install the APK (`./gradlew assembleDebug`).
4. **Zero Permissions:** The application operates completely offline without needing any internet (`INTERNET`) permission.

---

## 📄 License & Citation

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

If you use this repository or research in your work, please cite:
```bibtex
@software{mundari_nmt_2026,
  author = {Mundari NMT Project Team},
  title = {Offline Hindi-Mundari Neural Machine Translation System},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/your-username/mundari-nmt}
}
```
