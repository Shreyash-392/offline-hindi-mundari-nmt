import time
import os
import sys
import torch
import numpy as np
import onnxruntime as ort

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.transformer import TransformerNMT
from tokenizer.tokenizer import HindiMundariTokenizer


PAD_ID = 3
MAX_LENGTH = 80


def generate_pytorch(model, tokenizer, text, device):
    model.eval()
    src_ids = tokenizer.encode(text, max_length=MAX_LENGTH)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_padding_mask = src == PAD_ID

    generated = [tokenizer.bos_id]

    start_time = time.perf_counter()
    for _ in range(MAX_LENGTH - 1):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_padding_mask = tgt == PAD_ID
        tgt_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), device=device, dtype=torch.bool),
            diagonal=1,
        )

        with torch.no_grad():
            output = model(
                src,
                tgt,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask,
                tgt_mask=tgt_mask,
            )

        next_token = output[:, -1, :].argmax(dim=-1).item()
        generated.append(next_token)
        if next_token == tokenizer.eos_id:
            break

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return tokenizer.decode(generated), elapsed_ms


def generate_onnx(session, tokenizer, text):
    src_ids = tokenizer.encode(text, max_length=MAX_LENGTH)
    src_padded = src_ids + [PAD_ID] * (MAX_LENGTH - len(src_ids))
    src_np = np.array([src_padded], dtype=np.int64)
    src_padding_mask_np = src_np == PAD_ID

    generated = [tokenizer.bos_id]

    start_time = time.perf_counter()
    for _ in range(MAX_LENGTH - 1):
        tgt_padded = generated + [PAD_ID] * (MAX_LENGTH - len(generated))
        tgt_np = np.array([tgt_padded], dtype=np.int64)
        tgt_padding_mask_np = tgt_np == PAD_ID
        tgt_mask_np = np.triu(np.ones((MAX_LENGTH, MAX_LENGTH), dtype=bool), k=1)

        inputs = {
            "src": src_np,
            "tgt": tgt_np,
            "src_padding_mask": src_padding_mask_np,
            "tgt_padding_mask": tgt_padding_mask_np,
            "tgt_mask": tgt_mask_np,
        }

        outputs = session.run(None, inputs)
        logits = outputs[0]  # shape: (1, 80, vocab_size)

        curr_pos = len(generated) - 1
        next_token = int(np.argmax(logits[0, curr_pos, :]))
        generated.append(next_token)
        if next_token == tokenizer.eos_id:
            break

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return tokenizer.decode(generated), elapsed_ms


def benchmark_direction(dir_name, pt_checkpoint, onnx_fp32_path, onnx_int8_path, sample_text):
    print(f"\n============================================================")
    print(f"BENCHMARK & VERIFICATION: {dir_name}")
    print(f"============================================================")

    tokenizer = HindiMundariTokenizer()
    device = torch.device("cpu")

    # 1. PyTorch Model
    pt_model = TransformerNMT().to(device)
    pt_ckpt = torch.load(pt_checkpoint, map_location=device)
    pt_model.load_state_dict(pt_ckpt["model_state_dict"])
    pt_size_mb = os.path.getsize(pt_checkpoint) / (1024 * 1024)

    # 2. ONNX FP32
    onnx_fp32_sess = ort.InferenceSession(onnx_fp32_path, providers=["CPUExecutionProvider"])
    fp32_size_mb = os.path.getsize(onnx_fp32_path) / (1024 * 1024)

    # 3. ONNX INT8
    onnx_int8_sess = ort.InferenceSession(onnx_int8_path, providers=["CPUExecutionProvider"])
    int8_size_mb = os.path.getsize(onnx_int8_path) / (1024 * 1024)

    # Warmup runs
    generate_pytorch(pt_model, tokenizer, sample_text, device)
    generate_onnx(onnx_fp32_sess, tokenizer, sample_text)
    generate_onnx(onnx_int8_sess, tokenizer, sample_text)

    # Run Benchmark (5 iterations)
    pt_times, fp32_times, int8_times = [], [], []
    pt_out, fp32_out, int8_out = "", "", ""

    for _ in range(5):
        pt_out, t_pt = generate_pytorch(pt_model, tokenizer, sample_text, device)
        fp32_out, t_fp32 = generate_onnx(onnx_fp32_sess, tokenizer, sample_text)
        int8_out, t_int8 = generate_onnx(onnx_int8_sess, tokenizer, sample_text)

        pt_times.append(t_pt)
        fp32_times.append(t_fp32)
        int8_times.append(t_int8)

    avg_pt_ms = float(np.mean(pt_times))
    avg_fp32_ms = float(np.mean(fp32_times))
    avg_int8_ms = float(np.mean(int8_times))

    print(f"\nSample Input ({dir_name}): {sample_text}")
    print(f"PyTorch FP32 Output: {pt_out}")
    print(f"ONNX FP32 Output:    {fp32_out}")
    print(f"ONNX INT8 Output:    {int8_out}")

    print("\nModel Comparison Table:")
    print(f"{'Runtime Format':<20}{'File Size (MB)':>18}{'CPU Latency (ms)':>20}{'Size Ratio':>15}")
    print("-" * 75)
    print(f"{'PyTorch FP32':<20}{pt_size_mb:>18.2f}{avg_pt_ms:>20.2f}{'100%':>15}")
    print(f"{'ONNX FP32':<20}{fp32_size_mb:>18.2f}{avg_fp32_ms:>20.2f}{f'{100*fp32_size_mb/pt_size_mb:.1f}%':>15}")
    print(f"{'ONNX INT8':<20}{int8_size_mb:>18.2f}{avg_int8_ms:>20.2f}{f'{100*int8_size_mb/pt_size_mb:.1f}%':>15}")

    return {
        "pt_size": pt_size_mb,
        "fp32_size": fp32_size_mb,
        "int8_size": int8_size_mb,
        "pt_lat": avg_pt_ms,
        "fp32_lat": avg_fp32_ms,
        "int8_lat": avg_int8_ms,
    }


def main():
    # Benchmark Hindi -> Mundari
    hi_sample = "दो माह से वेतन नहीं मिला।"
    benchmark_direction(
        dir_name="Hindi -> Mundari",
        pt_checkpoint="checkpoints/transformer_best.pt",
        onnx_fp32_path="checkpoints/transformer_hindi_mundari.onnx",
        onnx_int8_path="checkpoints/transformer_hindi_mundari_int8.onnx",
        sample_text=hi_sample,
    )

    # Benchmark Mundari -> Hindi
    mun_sample = "बार चानडुः ते नाला का बोसाआकाना।"
    benchmark_direction(
        dir_name="Mundari -> Hindi",
        pt_checkpoint="checkpoints/transformer_mundari_hindi_best.pt",
        onnx_fp32_path="checkpoints/transformer_mundari_hindi.onnx",
        onnx_int8_path="checkpoints/transformer_mundari_hindi_int8.onnx",
        sample_text=mun_sample,
    )


if __name__ == "__main__":
    main()
