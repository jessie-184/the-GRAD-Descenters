import os, sys, math, json, time, glob, collections
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import tqdm 
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
import time, torch
from torch.amp import autocast
#from tqdm import tqdm
import jiwer


INPUT_DIR = Path('/data/users1/afani1/neural_seq_decoder/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final')

# get all hdf5 files recursively under the input dir
hdf5_files = sorted([p for p in INPUT_DIR.rglob("*.hdf5")])
print(f"Found {len(hdf5_files)} .hdf5 files.")
top_subdirs = [d for d in INPUT_DIR.iterdir() if d.is_dir()]

if len(hdf5_files) == 0:
    raise SystemExit("No .hdf5 files found.")

# choose a representative train file if possible (prefer names containing 'train' or 'data_train')
train_candidates = [p for p in hdf5_files if 'train' in p.name.lower()]

#---- printing sample dimensions -----
#sample_file = train_candidates[0] if train_candidates else hdf5_files[0]
#print("Using sample file for inspection:", sample_file.relative_to(INPUT_DIR))

#with h5py.File(sample_file, 'r') as hf:
#    print("Top-level keys:", list(hf.keys()))
#    # iterate and print shapes/types for datasets and groups
#    def print_group(g, indent=0):
#        for k in g:
#            item = g[k]
#            if isinstance(item, h5py.Dataset):
#                print("  " * indent + f"- Dataset: {k}  shape={item.shape}  dtype={item.dtype}")
#            elif isinstance(item, h5py.Group):
#                print("  " * indent + f"- Group: {k}")
#                print_group(item, indent+1)
#    print_group(hf)

all_h5 = sorted([p for p in INPUT_DIR.rglob("data_train.hdf5")])
print("Found train hdf5 files:", len(all_h5))

val_h5 = sorted([p for p in INPUT_DIR.rglob("data_val.hdf5")])
print("Found val hdf5 files:", len(val_h5))

#---- slot files into a df ----
rows = []
for h5path in tqdm.tqdm(all_h5, desc="Scanning train hdf5 files"):
    with h5py.File(h5path, 'r') as hf:
        for grp_name in hf.keys():
            if grp_name.startswith('trial_'):
                # optional: verify it has input_features and transcription/seq_class_ids
                grp = hf[grp_name]
                if 'input_features' in grp:
                    try:
                        tshape = grp['input_features'].shape
                    except Exception:
                        tshape = None
                    rows.append({
                        'h5_path': str(h5path),
                        'group': grp_name,
                        'feat_shape': tshape
                    })

idx_df = pd.DataFrame(rows)
print("Total trials indexed:", len(idx_df))
print(idx_df.head())

val_rows = []
for h5path in val_h5:
    with h5py.File(h5path, 'r') as hf:
        for grp_name in hf.keys():
            if grp_name.startswith('trial_') and 'input_features' in hf[grp_name]:
                val_rows.append({'h5_path': str(h5path), 'group': grp_name})
val_df = pd.DataFrame(val_rows)
print("Total val trials indexed:", len(val_df))

train_df_index = idx_df  # from earlier cell
print("Train trials:", len(train_df_index))

#---- function and class defintions ----
def collate_for_ctc(batch):
    """
    batch: list of tuples (feats_tensor [T,C], target_tensor [L])
    Returns:
      x_padded: [B, C, T_max]
      targets_concat: 1D tensor of concatenated targets
      input_lengths: tensor [B] lengths (in frames, i.e., T_i)
      target_lengths: tensor [B] lengths (L_i)
    """
    xs, ys = zip(*batch)
    x_lens = [x.shape[0] for x in xs]
    t_lens = [y.shape[0] for y in ys]

    # pad inputs along time to max_t
    max_t = max(x_lens)
    channels = xs[0].shape[1]
    x_padded = torch.zeros(len(xs), channels, max_t, dtype=torch.float32)
    for i, x in enumerate(xs):
        T = x.shape[0]
        # x is [T, C] -> convert to [C, T]
        x_padded[i, :, :T] = x.permute(1,0)

    # concatenate targets to 1D for CTC
    if sum(t_lens) > 0:
        targets_concat = torch.cat([y.to(torch.long) for y in ys])
    else:
        targets_concat = torch.tensor([], dtype=torch.long)

    return x_padded, targets_concat, torch.tensor(x_lens, dtype=torch.long), torch.tensor(t_lens, dtype=torch.long)

class BrainDataset(Dataset):
    def __init__(self, index_df, cache_size=8, max_len=None):
        self.df = index_df.reset_index(drop=True)
        self.max_len = max_len
        self._cache_size = cache_size
        self._file_cache = collections.OrderedDict()

    def __len__(self):
        return len(self.df)

    def _open_file(self, path):
        if path in self._file_cache:
            self._file_cache.move_to_end(path)
            return self._file_cache[path]
        f = h5py.File(path, 'r')
        self._file_cache[path] = f
        if len(self._file_cache) > self._cache_size:
            old_path, old_f = self._file_cache.popitem(last=False)
            try: old_f.close()
            except: pass
        return f

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        f = self._open_file(row['h5_path'])
        g = f[row['group']]

        feats = g['input_features'][()].astype('float32')
        if 'transcription' in g:
            tgt = g['transcription'][()]
        elif 'seq_class_ids' in g:
            tgt = g['seq_class_ids'][()]
        else:
            raise KeyError(f"No target found in {row['h5_path']}::{row['group']}")

        tgt = np.array(tgt, dtype='int64').reshape(-1)
        if tgt.shape[0] >= 64:
            nz = np.nonzero(tgt)[0]
            tgt = tgt[:nz[-1]+1] if nz.size else tgt[:1]

        if self.max_len and feats.shape[0] > self.max_len:
            start = (feats.shape[0] - self.max_len)//2
            feats = feats[start:start+self.max_len]

        return torch.from_numpy(feats), torch.from_numpy(tgt)

    def close(self):
        for _, f in list(self._file_cache.items()):
            try: f.close()
            except: pass
        self._file_cache.clear()


#---- creating dataset objects ----
train_ds = BrainDataset(train_df_index, cache_size=12, max_len=None)
val_ds = BrainDataset(val_df, cache_size=4, max_len=None)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_for_ctc, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_for_ctc, num_workers=2)

# sanity-check: load one batch
batch = next(iter(train_loader))
x_padded, targets_concat, input_lengths, target_lengths = batch
print("x_padded shape (B,C,T):", x_padded.shape)
print("targets_concat shape (sumL,):", targets_concat.shape)
print("input_lengths:", input_lengths)
print("target_lengths:", target_lengths)

train_ds.close()
val_ds.close()


#---- defining the transformer ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [B, T, D]
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return x

class ConvStem(nn.Module):
    def __init__(self, in_ch, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

    def forward(self, x):
        # input: [B, C, T]
        return self.net(x)  # [B, D, T']

class BrainToTextModel(nn.Module):
    def __init__(self, in_ch=512, d_model=384, nhead=8, num_layers=6, vocab_size=200):
        super().__init__()
        self.conv = ConvStem(in_ch, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model * 4, dropout=0.1, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x):
        # x: [B, C, T]
        x = self.conv(x)          # [B, D, T']
        x = x.permute(0, 2, 1)    # [B, T', D]
        x = self.pos_enc(x)
        x = x.permute(1, 0, 2)    # [T', B, D] (required by Transformer)
        x = self.transformer(x)   # [T', B, D]
        x = x.permute(1, 0, 2)    # [B, T', D]
        logits = self.fc(x)       # [B, T', vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

#----initialize model, optimizer, lrscheduler, and loss----
sample_batch = next(iter(train_loader))
x_padded, targets_concat, input_lengths, target_lengths = sample_batch
in_channels = x_padded.shape[1]
print("Detected input channels:", in_channels)

vocab_size_estimate = 256  # since labels are already integer-encoded
model = BrainToTextModel(in_ch=in_channels, d_model=384, nhead=8, num_layers=6, vocab_size=vocab_size_estimate).to('cuda')

ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = AdamW(model.parameters(), lr=3e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
scaler = GradScaler()

print("Model initialized successfully.")
print("Total parameters:", sum(p.numel() for p in model.parameters())/1e6, "Million")


#---- training and validation loop ----
def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch, device='cuda'):
    model.train()
    total_loss, steps = 0.0, 0
    start_time = time.time()

    for batch in tqdm.tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False):
        x, targets_concat, input_lengths, target_lengths = batch
        x, targets_concat = x.to(device), targets_concat.to(device)
        input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)

        # 🔧 Fix for ConvStem downsampling (stride 2 twice → /4)
        input_lengths = torch.div(input_lengths, 4, rounding_mode='floor')

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', dtype=torch.float16):
            log_probs = model(x)                     # [B, T', V]
            log_probs = log_probs.permute(1, 0, 2)   # [T', B, V] for CTC
            loss = ctc_loss(log_probs, targets_concat, input_lengths, target_lengths)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        steps += 1

    avg_loss = total_loss / max(1, steps)
    print(f"Epoch {epoch}: train loss = {avg_loss:.4f}  (time {time.time() - start_time:.1f}s)")
    return avg_loss

@torch.no_grad()
def validate(model, loader, epoch, device='cuda'):
    model.eval()
    total_loss, steps = 0.0, 0
    preds, refs = [], []

    for batch in tqdm.tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False):
        x, targets_concat, input_lengths, target_lengths = batch
        x, targets_concat = x.to(device), targets_concat.to(device)
        input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)

        # 🔧 Adjust lengths for ConvStem downsampling
        input_lengths = torch.div(input_lengths, 4, rounding_mode='floor')

        log_probs = model(x)
        log_probs = log_probs.permute(1, 0, 2)
        loss = ctc_loss(log_probs, targets_concat, input_lengths, target_lengths)
        total_loss += loss.item()
        steps += 1

        # --- Greedy decode for approximate WER ---
        decoded_batch = log_probs.argmax(-1).permute(1, 0).cpu().numpy()
        idx = 0
        for i, L in enumerate(target_lengths):
            true_seq = targets_concat[idx:idx + L].cpu().numpy().tolist()
            idx += L
            pred_seq = decoded_batch[i]
            # collapse repeats + remove blanks
            pred_seq_clean = [p for j, p in enumerate(pred_seq) if (j == 0 or p != pred_seq[j - 1]) and p != 0]
            ref_seq_clean = [p for p in true_seq if p != 0]
            preds.append(" ".join(map(str, pred_seq_clean)))
            refs.append(" ".join(map(str, ref_seq_clean)))

    val_loss = total_loss / max(1, steps)
    try:
        wer_score = jiwer.wer(refs, preds)
    except Exception:
        wer_score = None

    print(f"Epoch {epoch}: val loss = {val_loss:.4f},  WER ≈ {wer_score if wer_score else 'N/A'}")
    return val_loss, wer_score

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(ckpt, path)
    print(f"✅ Saved checkpoint: {path}")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 40  # test run first
best_val_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, device=device)
    val_loss, val_wer = validate(model, val_loader, epoch, device=device)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, scheduler, epoch, f"./model_best_epoch{epoch}.pt")

print("Training complete ✅")

@torch.no_grad()
def greedy_decode(log_probs):
    """
    Performs greedy decoding of log probabilities.
    Removes blanks (0) and repeated tokens.
    Args:
        log_probs: [B, T', V] tensor
    Returns:
        List of integer token sequences
    """
    preds = log_probs.argmax(-1).cpu().numpy()  # [B, T']
    decoded = []
    for i in range(preds.shape[0]):
        seq = preds[i]
        seq_clean = [p for j, p in enumerate(seq)
                     if (j == 0 or p != seq[j - 1]) and p != 0]
        decoded.append(seq_clean)
    return decoded

@torch.no_grad()
def infer_file(model, path, device="cuda"):
    """
    Runs inference on all trials inside one HDF5 test file.
    Args:
        model: trained BrainToTextModel
        path: path to .hdf5 file
        device: 'cuda' or 'cpu'
    Returns:
        List of dicts: [{'id': '<filename>_<trial>', 'transcript': '<decoded tokens>'}, ...]
    """
    preds_list = []
    with h5py.File(path, "r") as hf:
        for trial in hf.keys():
            if "input_features" not in hf[trial]:
                continue

            feats = hf[trial]["input_features"][()].astype("float32")
            x = torch.from_numpy(feats).unsqueeze(0).permute(0, 2, 1).to(device)  # [1, C, T]

            with autocast("cuda", dtype=torch.float16):
                log_probs = model(x)  # [1, T', V]

            decoded = greedy_decode(log_probs)[0]
            decoded_str = " ".join(map(str, decoded))

            preds_list.append({
                "id": f"{Path(path).stem}_{trial}",
                "transcript": decoded_str
            })

    return preds_list

import random
val_files = sorted([p for p in INPUT_DIR.rglob("data_val.hdf5")])
print(f"Found {len(val_files)} validation files.")
sample_files = random.sample(val_files, 3)  # show samples from 3 val files

for val_path in sample_files:
    print(f"\n📘 File: {val_path.name}")
    with h5py.File(val_path, "r") as hf:
        # Pick 2 random trials from each file
        trial_names = random.sample(list(hf.keys()), 2)
        for trial in trial_names:
            if "input_features" not in hf[trial]:
                continue
            feats = hf[trial]["input_features"][()].astype("float32")
            transcript = hf[trial]["transcription"][()].astype("int32")

            # Decode transcript (true)
            true_text = " ".join(map(str, transcript.tolist()))

            # Model prediction
            x = torch.from_numpy(feats).unsqueeze(0).permute(0, 2, 1).to(device)
            with autocast("cuda", dtype=torch.float16):
                log_probs = model(x)
            decoded_seq = greedy_decode(log_probs)[0]
            pred_text = " ".join(map(str, decoded_seq))

            print(f"🧠 Trial: {trial}")
            print(f"   ➤ Predicted: {pred_text[:200]}...")
            print(f"   ➤ Actual:    {true_text[:200]}...\n")

def decode_ascii(int_seq):
    """Convert list/array of integers to readable text (ASCII decoding)."""
    chars = [chr(i) for i in int_seq if 32 <= i <= 126]  # printable characters
    return "".join(chars)

# Display a few samples again in readable form
val_path = random.choice(val_files)
print(f"\n📘 File: {val_path.name}")

with h5py.File(val_path, "r") as hf:
    trial_names = random.sample(list(hf.keys()), 3)
    for trial in trial_names:
        if "input_features" not in hf[trial]:
            continue

        feats = hf[trial]["input_features"][()].astype("float32")
        transcript = hf[trial]["transcription"][()].astype("int32")

        # Run prediction
        x = torch.from_numpy(feats).unsqueeze(0).permute(0, 2, 1).to(device)
        with autocast("cuda", dtype=torch.float16):
            log_probs = model(x)
        decoded_seq = greedy_decode(log_probs)[0]

        # Decode to readable text
        pred_text = decode_ascii(decoded_seq)
        true_text = decode_ascii(transcript.tolist())

        print(f"🧠 Trial: {trial}")
        print(f"   ➤ Predicted: {pred_text}")
        print(f"   ➤ Actual:    {true_text}\n")


