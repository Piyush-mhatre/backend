"""
Run this ONCE, locally (you already have torch + transformers installed
there from before). This replaces the torch-based quantized model with
an ONNX one, quantized the same way (dynamic INT8) — but at inference
time on Render, we no longer need torch or transformers' modeling
classes at all, just onnxruntime, which is roughly 10x lighter to import
than torch. That's the actual fix for the persistent OOM: torch's own
runtime (libtorch, MKL, OpenMP) costs 200-300MB just by existing in the
process, regardless of model size, and that's what's been eating
Render's 512MB free tier even after all the quantization/loading fixes.

Since this app never fine-tuned FinBERT (confirmed — it's the base
ProsusAI/finbert, just quantized), we export directly from the
Hugging Face model. No custom weights to worry about losing.

Outputs (all under models/):
  models/finbert_int8.onnx   <- upload this as the new GitHub Release asset
  models/tokenizer/          <- small files (~1-2MB total), commit these
                                 directly to git, no Release needed
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_NAME = "ProsusAI/finbert"
OUT_DIR = "models"
FP32_ONNX_PATH = os.path.join(OUT_DIR, "finbert_fp32_temp.onnx")  # deleted at the end
INT8_ONNX_PATH = os.path.join(OUT_DIR, "finbert_int8.onnx")
TOKENIZER_DIR = os.path.join(OUT_DIR, "tokenizer")

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading tokenizer + base FinBERT model from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

print(f"Saving tokenizer files to {TOKENIZER_DIR}/ (commit these to git directly)...")
tokenizer.save_pretrained(TOKENIZER_DIR)

print("Exporting to ONNX (fp32)...")
dummy = tokenizer(
    "Sample financial news sentence for ONNX export tracing.",
    max_length=64,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    FP32_ONNX_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=14,
)

print("Quantizing ONNX model to INT8 (dynamic quantization)...")
quantize_dynamic(
    model_input=FP32_ONNX_PATH,
    model_output=INT8_ONNX_PATH,
    weight_type=QuantType.QInt8,
)

os.remove(FP32_ONNX_PATH)

size_mb = os.path.getsize(INT8_ONNX_PATH) / (1024 * 1024)
print(f"\nDone. {INT8_ONNX_PATH} is {size_mb:.1f} MB.")
print("Next steps:")
print(f"  1. Upload {INT8_ONNX_PATH} as a new GitHub Release asset")
print(f"  2. git add {TOKENIZER_DIR} and commit+push it (small, no Release needed)")
print("  3. Update Render's Build Command to curl finbert_int8.onnx instead of the .pt file")
print("  4. Remove torch from requirements.txt, add onnxruntime")
