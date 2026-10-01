# Offline Android Mobile Deployment Architecture

## 1. Architecture Overview
The Android application ([`android/`](file:///d:/ML_engine/SantaliEngine/android/)) runs completely offline without requiring cellular data, Wi-Fi, or remote cloud API servers.

```
                      ┌─────────────────────────────────┐
                      │    Android App User Interface   │
                      │    (Kotlin MainActivity)        │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │  Local Tokenizer (Tokenizer.kt) │
                      │  (SentencePiece BPE 8K Vocab)   │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │  Local ONNX Runtime Mobile SDK  │
                      │  (TranslationEngine.kt)         │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │  INT8 Quantized Model Assets    │
                      │  (transformer_*_int8.onnx)      │
                      └─────────────────────────────────┘
```

---

## 2. Key Android Implementation Specs
- **Min SDK Version:** Android 7.0 (API Level 24)
- **Target SDK Version:** Android 14 (API Level 34)
- **Local Assets:**
  - `app/src/main/assets/hindi_mundari.vocab` (Vocabulary dictionary)
  - `app/src/main/assets/transformer_hindi_mundari_int8.onnx` (Hindi → Mundari model)
  - `app/src/main/assets/transformer_mundari_hindi_int8.onnx` (Mundari → Hindi model)
- **Execution Dependencies:** `com.microsoft.onnxruntime:onnxruntime-android:1.17.0`
- **Network Permissions:** **None** (`0` network sockets opened during translation).
