import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from jiwer import wer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load DeepSeek 1.5B
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    
)
model.eval()


jsonl_path = Path(r"ctc_checkpoints/ctc_for_llm.jsonl")  # note r"" to be safe on Windows
preds = []
truths = []
ids = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        # match your JSON structure:
        # {"id": ..., "ctc": "...", "ref": "..."}
        ids.append(obj["id"])
        preds.append(obj["ctc"])   # CTC output
        truths.append(obj["ref"])  # reference / ground truth

llm_outputs = []

for i, text in enumerate(preds):
    prompt = (
        "You are an ASR error corrector. Fix the transcription.\n"
        f"Input: {text}\n"
        "Corrected: "
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
    )
    corrected = tokenizer.decode(out[0], skip_special_tokens=True)
    corrected = corrected.split("Corrected:")[-1].strip()

    if i < 10:
        print("CTC:", text)
        print("LLM:", corrected)
        print("TRUE:", truths[i])
        print("-" * 60)

    llm_outputs.append(corrected)

# Save predictions as numpy file for your pipeline
np.save("deepseek_predictions.npy", np.array(llm_outputs))

# Compute WER
wer_score = wer(truths, llm_outputs)
print("\n===== FINAL WER =====")
print("DeepSeek-1.5B WER:", wer_score)
