import numpy as np
from pathlib import Path
import json
import random
import math
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# CONFIG
# ==========================================================

DATA_DIR = Path(r"C:\Users\morga\LM\formatted_data")  # change if needed
OUT_DIR = Path(r"C:\Users\morga\LM\ctc_checkpoints")
OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 10        # start small; you can increase later
HIDDEN_DIM = 256
NUM_LAYERS = 3
LR = 1e-3
VAL_SPLIT = 0.1        # 10% data for validation

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# 1) LOAD DATA
# ==========================================================

decoder_inputs_raw = np.load(DATA_DIR / "decoder_inputs.npy", allow_pickle=True)
transcripts_raw = np.load(DATA_DIR / "transcripts.npy", allow_pickle=True)

print("decoder_inputs.npy type/shape:", type(decoder_inputs_raw), getattr(decoder_inputs_raw, "shape", None))
print("transcripts.npy type/shape:", type(transcripts_raw), getattr(transcripts_raw, "shape", None))

# Convert to python lists of numpy arrays [T_i, D] and label arrays [L_i]
features_list: List[np.ndarray] = []
labels_list_raw: List[np.ndarray] = []

# handle object arrays or normal ndarrays
if decoder_inputs_raw.dtype == object:
    for x in decoder_inputs_raw:
        arr = np.asarray(x, dtype=np.float32)
        features_list.append(arr)
else:
    # assume shape [N, T, D]
    for i in range(decoder_inputs_raw.shape[0]):
        arr = decoder_inputs_raw[i].astype(np.float32)
        features_list.append(arr)

if transcripts_raw.dtype == object:
    for y in transcripts_raw:
        arr = np.asarray(y, dtype=np.int64).reshape(-1)
        labels_list_raw.append(arr)
else:
    # assume shape [N, L] with padding; strip trailing zeros
    for i in range(transcripts_raw.shape[0]):
        arr = np.asarray(transcripts_raw[i], dtype=np.int64)
        nz = np.nonzero(arr)[0]
        if nz.size > 0:
            arr = arr[: nz[-1] + 1]
        labels_list_raw.append(arr)

num_samples = len(features_list)
print(f"Loaded {num_samples} samples.")

# ==========================================================
# 2) BUILD VOCABULARY (map original ints -> compact indices)
# ==========================================================

all_symbols = sorted({int(v) for seq in labels_list_raw for v in seq.tolist()})
print("Number of unique label symbols:", len(all_symbols))
print("First 20 symbols:", all_symbols[:20])

# 0 is reserved for CTC blank
symbol_to_idx: Dict[int, int] = {sym: i + 1 for i, sym in enumerate(all_symbols)}
idx_to_symbol: Dict[int, int] = {i + 1: sym for i, sym in enumerate(all_symbols)}

vocab_size = len(all_symbols) + 1  # +1 for blank
print("CTC vocab_size (including blank=0):", vocab_size)

# convert labels
labels_list: List[np.ndarray] = []
for seq in labels_list_raw:
    mapped = np.array([symbol_to_idx[int(v)] for v in seq], dtype=np.int64)
    labels_list.append(mapped)

# save mapping for later decoding
with open(OUT_DIR / "vocab_mapping.json", "w") as f:
    json.dump(
        {
            "symbol_to_idx": symbol_to_idx,
            "idx_to_symbol": idx_to_symbol,
            "vocab_size": vocab_size,
        },
        f,
        indent=2,
    )
print("Saved vocab_mapping.json")

# ==========================================================
# 3) TRAIN / VAL SPLIT
# ==========================================================

indices = list(range(num_samples))
random.shuffle(indices)
split = int(num_samples * (1 - VAL_SPLIT))
train_idx = indices[:split]
val_idx = indices[split:]

print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

# determine input dimension
input_dim = features_list[0].shape[1]
print("Input feature dimension D =", input_dim)

# ==========================================================
# 4) DATASET & COLLATE
# ==========================================================

class NpyBrainDataset(Dataset):
    def __init__(self, feats_list, labels_list, index_list):
        self.feats = [feats_list[i] for i in index_list]
        self.labels = [labels_list[i] for i in index_list]

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        x = self.feats[idx]   # [T, D]
        y = self.labels[idx]  # [L]
        return torch.from_numpy(x), torch.from_numpy(y)

def collate_for_ctc(batch):
    xs, ys = zip(*batch)
    lengths = [x.shape[0] for x in xs]      # T_i
    t_lengths = [y.shape[0] for y in ys]    # L_i

    max_t = max(lengths)
    dim = xs[0].shape[1]
    B = len(xs)

    x_padded = torch.zeros(B, dim, max_t, dtype=torch.float32)
    for i, x in enumerate(xs):
        T = x.shape[0]
        x_padded[i, :, :T] = x.permute(1, 0)

    targets_concat = torch.cat([y.to(torch.long) for y in ys])

    return (
        x_padded,                      # [B, D, T_max]
        targets_concat,                # [sum L_i]
        torch.tensor(lengths, dtype=torch.long),   # input_lengths
        torch.tensor(t_lengths, dtype=torch.long), # target_lengths
    )

train_ds = NpyBrainDataset(features_list, labels_list, train_idx)
val_ds = NpyBrainDataset(features_list, labels_list, val_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_for_ctc, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_for_ctc, num_workers=0)

# ==========================================================
# 5) MODEL: BiGRU encoder + CTC
# ==========================================================

class CTCEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, vocab_size):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        # x: [B, D, T]
        x = x.permute(0, 2, 1)    # [B, T, D]
        x = self.proj(x)          # [B, T, H]
        out, _ = self.rnn(x)      # [B, T, 2H]
        logits = self.fc(out)     # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs          # [B, T, V]

model = CTCEncoder(input_dim, HIDDEN_DIM, NUM_LAYERS, vocab_size).to(device)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Model params (M):", sum(p.numel() for p in model.parameters()) / 1e6)

# ==========================================================
# 6) TRAINING + VALIDATION
# ==========================================================

def greedy_decode_ctc(log_probs):
    """
    log_probs: [B, T, V]
    returns list of sequences (list[int]) in vocab indices
    """
    preds = log_probs.argmax(-1).cpu().numpy()  # [B, T]
    decoded = []
    for seq in preds:
        out = []
        prev = None
        for p in seq:
            if p != 0 and p != prev:
                out.append(int(p))
            prev = p
        decoded.append(out)
    return decoded

def train_one_epoch(epoch):
    model.train()
    total_loss, steps = 0.0, 0

    for x, targets_concat, input_lengths, target_lengths in train_loader:
        x = x.to(device)
        targets_concat = targets_concat.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        log_probs = model(x)              # [B, T, V]
        log_probs_t = log_probs.permute(1, 0, 2)   # [T, B, V]
        loss = ctc_loss(log_probs_t, targets_concat, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    avg = total_loss / max(1, steps)
    print(f"Epoch {epoch}: train loss = {avg:.4f}")
    return avg

@torch.no_grad()
def validate(epoch):
    model.eval()
    total_loss, steps = 0.0, 0

    for x, targets_concat, input_lengths, target_lengths in val_loader:
        x = x.to(device)
        targets_concat = targets_concat.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        log_probs = model(x)
        log_probs_t = log_probs.permute(1, 0, 2)
        loss = ctc_loss(log_probs_t, targets_concat, input_lengths, target_lengths)

        total_loss += loss.item()
        steps += 1

    avg = total_loss / max(1, steps)
    print(f"Epoch {epoch}: val loss = {avg:.4f}")
    return avg

best_val = float("inf")
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(epoch)
    val_loss = validate(epoch)

    if val_loss < best_val:
        best_val = val_loss
        ckpt_path = OUT_DIR / "ctc_model_best.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "vocab_size": vocab_size,
            },
            ckpt_path,
        )
        print(f"✅ Saved new best checkpoint -> {ckpt_path}")

print("Training finished. Best val loss:", best_val)
