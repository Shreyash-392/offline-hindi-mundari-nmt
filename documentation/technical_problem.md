# Technical Problem Statement — Offline Low-Resource NMT

## 1. Context & Technical Challenges
Mundari is a severely low-resource Austroasiatic language spoken primarily in eastern India. Machine translation for Mundari faces three fundamental technical bottlenecks:

1. **Extreme Data Scarcity & Domain Noise:**
   Unlike high-resource languages with millions of parallel sentences, parallel Hindi–Mundari data is limited to ~30,000 sentence pairs. Standard large translation architectures (e.g. 6-layer 512-dim Transformers or 25M+ parameter models) suffer from severe over-fitting, validation divergence, and output degeneracy.

2. **Subword Out-of-Vocabulary (OOV) Splitting:**
   Agglutinative morphology in Mundari produces complex word forms. Word-level tokenization leads to massive vocabulary explosion and rare word failure.

3. **Edge Resource & Connectivity Constraints:**
   Mundari-speaking regions frequently lack reliable high-speed mobile internet. Existing cloud translation APIs cannot operate in offline rural scenarios. Furthermore, mobile devices possess strict memory (RAM) and CPU battery thermal budgets, making 100MB+ unquantized PyTorch models impractical.

---

## 2. Quantitative Problem Definition
- **Data Budget:** Limited strictly to 30,225 preprocessed parallel pairs.
- **Model Size Limit:** Mobile APK asset budget requires total model footprint under **30 MB**.
- **Latency Budget:** Inference latency per sentence must be under **500 ms** on edge ARM CPUs.
- **Accuracy Criteria:** Translation fidelity must exceed baseline non-masked collapse models without sentence repetition or empty string generation.
