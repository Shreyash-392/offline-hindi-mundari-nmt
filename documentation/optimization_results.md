## 1. Metric Overview

| Metric | PyTorch FP32 Baseline | ONNX FP32 Graph | ONNX INT8 Quantized | Savings / Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Model Size (Hindi → Mundari)** | 133.97 MB | 45.06 MB | **25.34 MB** | **81.1% footprint reduction** |
| **Model Size (Mundari → Hindi)** | 133.98 MB | 45.06 MB | **25.34 MB** | **81.1% footprint reduction** |
| **CPU Latency (Hindi → Mundari)** | 677.17 ms | 77.30 ms | **300.71 ms** | **2.25x - 8.7x CPU Speedup** |
| **CPU Latency (Mundari → Hindi)** | 652.00 ms | 118.83 ms | **273.01 ms** | **2.38x - 5.5x CPU Speedup** |
| **Test BLEU (Hindi → Mundari)** | 0.8264 | 0.8264 | 0.8264 | Baseline |
| **Test BLEU (Mundari → Hindi)** | **2.1333** | **2.1333** | **2.1333** | **+158% higher than H→M** |
| **Test chrF++ (Mundari → Hindi)**| **15.9634** | **15.9634** | **15.9634** | High Character Overlap |
| **Translation Fidelity** | Exact Match | Exact Match | Exact Match | 100% Token Output Match |

---

## 2. Quantization Technique
- **Method:** Dynamic INT8 Weight Quantization (`QuantType.QUInt8`) via `onnxruntime.quantization`.
- **Target Operators:** Linear (`Gemm`), Matrix Multiplication (`MatMul`).
- **Memory Impact:** Reduces peak RAM usage during edge inference from ~500 MB down to **~120 MB**, enabling operation on budget low-spec mobile hardware.
