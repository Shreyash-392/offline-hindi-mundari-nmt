import os
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_model(input_onnx_path, output_onnx_path, model_name):
    print(f"\n--- Quantizing {model_name} ONNX Model to INT8 ---")

    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=output_onnx_path,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    orig_size = os.path.getsize(input_onnx_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_onnx_path) / (1024 * 1024)
    compression = (1 - (quant_size / orig_size)) * 100

    print(f"FP32 ONNX Model Size: {orig_size:.2f} MB")
    print(f"INT8 ONNX Model Size: {quant_size:.2f} MB")
    print(f"Size Reduction: {compression:.2f}%")


def main():
    # 1. Hindi -> Mundari (Augmented Model)
    quantize_model(
        input_onnx_path="checkpoints/transformer_hindi_mundari.onnx",
        output_onnx_path="checkpoints/transformer_hindi_mundari_int8.onnx",
        model_name="Hindi -> Mundari (Augmented)",
    )

    # 2. Mundari -> Hindi
    quantize_model(
        input_onnx_path="checkpoints/transformer_mundari_hindi.onnx",
        output_onnx_path="checkpoints/transformer_mundari_hindi_int8.onnx",
        model_name="Mundari -> Hindi",
    )


if __name__ == "__main__":
    main()
