import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# CONFIG
# ==========================================================

DATA_DIR = Path(r"C:\Users\morga\LM\formatted_data")
CKPT_DIR = Path(r"C:\Users\morga\LM\ctc_checkpoints")
CKPT_PATH = CKPT_DIR / "ctc_model_best.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# LOAD DATA
# ==========================================================

decoder_inputs_raw = np.load(DATA_DIR / "decoder_inputs.npy", allow_pickle=True)
transcripts_raw = np.load(DATA_DIR / "transcripts.npy", allow_pickle=True)

# normalize features to list of [T,D]
features_list = []
if decoder_inputs_raw.dtype == object:
    for x in decoder_inputs_raw:
        arr = np.asarray(x, dtype=np.float32)
        features_list.append(arr)
else:
    for i in range(decoder_inputs_raw.shape[0]):
        arr = decoder_inputs_raw[i].astype(np.float32)
        features_list.append(arr)

num_samples = len(features_list)
print("Total samples:", num_samples)
print(f"First feature shape: {features_list[0].shape}")  # DEBUG

# ==========================================================
# LOAD VOCAB / MODEL
# ==========================================================

vocab_path = CKPT_DIR / "vocab_mapping.json"
if not vocab_path.exists():
    raise FileNotFoundError(f"Vocab mapping not found at {vocab_path}")

with open(vocab_path, "r") as f:
    vocab_info = json.load(f)

# Fix: ensure proper type conversion
idx_to_symbol = {int(k): int(v) for k, v in vocab_info["idx_to_symbol"].items()}
vocab_size = vocab_info["vocab_size"]

print(f"Vocab size: {vocab_size}")
print(f"Sample vocab mapping: {list(idx_to_symbol.items())[:5]}")  # DEBUG

# Check if checkpoint exists
if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found at {CKPT_PATH}")

ckpt = torch.load(CKPT_PATH, map_location=device)
input_dim = ckpt["input_dim"]
hidden_dim = ckpt["hidden_dim"]
num_layers = ckpt["num_layers"]

print(f"Model config: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")

class CTCEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, vocab_size):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        # x: [B, D, T]
        x = x.permute(0, 2, 1)    # [B, T, D]
        x = self.proj(x)
        out, _ = self.rnn(x)
        logits = self.fc(out)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs  # [B,T,V]

model = CTCEncoder(input_dim, hidden_dim, num_layers, vocab_size).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("✓ Loaded model checkpoint from", CKPT_PATH)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def greedy_decode_ctc(log_probs):
    """Greedy CTC decoding: remove blanks and repetitions."""
    preds = log_probs.argmax(-1).cpu().numpy()  # [B,T]
    decoded = []
    for seq in preds:
        out = []
        prev = None
        for p in seq:
            if p != 0 and p != prev:  # 0 is blank
                out.append(int(p))
            prev = p
        decoded.append(out)
    return decoded

def seq_idx_to_ascii(idx_seq):
    """Convert sequence of vocab indices back to readable ASCII text."""
    try:
        symbols = [idx_to_symbol[i] for i in idx_seq if i in idx_to_symbol]
        chars = [chr(s) for s in symbols if 32 <= s <= 126]
        return "".join(chars)
    except Exception as e:
        print(f"Warning: Error converting sequence {idx_seq[:10]}... to text: {e}")
        return ""

# ==========================================================
# INFERENCE
# ==========================================================

pred_texts = []

print("\n" + "="*60)
print("RUNNING INFERENCE...")
print("="*60)

with torch.no_grad():
    for i, feats in enumerate(features_list):
        # Debug: check feature shape
        if i == 0:
            print(f"Processing sample {i}: input shape = {feats.shape}")
        
        # feats is [T, D], need [1, D, T] for model
        x = torch.from_numpy(feats).unsqueeze(0).permute(0, 2, 1).to(device)
        
        if i == 0:
            print(f"After transform: {x.shape} (expected [1, D, T])")
        
        log_probs = model(x)  # [1,T,V]
        
        if i == 0:
            print(f"Model output shape: {log_probs.shape}")
        
        decoded_idx_seq = greedy_decode_ctc(log_probs)[0]
        
        if i == 0:
            print(f"Decoded indices (first 20): {decoded_idx_seq[:20]}")
        
        text = seq_idx_to_ascii(decoded_idx_seq)
        pred_texts.append(text)

        # Show first 10 predictions with ground truth
        if i < 10:
            gt_item = transcripts_raw[i]
            if isinstance(gt_item, str):
                gt_text = gt_item
            else:
                gt_arr = np.asarray(gt_item, dtype=np.int64).reshape(-1)
                gt_text = "".join(chr(v) for v in gt_arr if 32 <= v <= 126)

            print(f"\n{'─'*60}")
            print(f"Sample {i}:")
            print(f"  PRED: '{text}'")
            print(f"  TRUE: '{gt_text}'")
            if len(decoded_idx_seq) > 0:
                print(f"  Decoded length: {len(decoded_idx_seq)} symbols")

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{num_samples} samples...")

# ==========================================================
# SAVE RESULTS
# ==========================================================

import csv
out_csv = CKPT_DIR / "ctc_predictions.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "prediction", "length"])
    for i, t in enumerate(pred_texts):
        writer.writerow([i, t, len(t)])

print("\n" + "="*60)
print(f"✓ Saved {len(pred_texts)} predictions to {out_csv}")
print("="*60)

# Calculate some statistics
non_empty = sum(1 for t in pred_texts if len(t) > 0)
avg_len = sum(len(t) for t in pred_texts) / len(pred_texts)
print(f"\nStats:")
print(f"  Non-empty predictions: {non_empty}/{len(pred_texts)}")
print(f"  Average prediction length: {avg_len:.1f} characters")