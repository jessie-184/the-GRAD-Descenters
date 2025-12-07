from pathlib import Path
import numpy as np
import h5py
import json

# ====== CONFIG ======
BASE_DIR = Path(
    r"C:\Users\morga\.cache\kagglehub\datasets\paulnkamau\brain-to-text-25-data\versions\3"
    r"\t15_copyTask_neuralData\hdf5_data_final"
)
OUT_DIR = Path(r"C:\Users\morga\LM\formatted_data")
OUT_DIR.mkdir(exist_ok=True)

print("Base dir:", BASE_DIR)
print("Output dir:", OUT_DIR)

# ====== COLLECT FILES ======
train_files = sorted(BASE_DIR.rglob("data_train.hdf5"))
val_files   = sorted(BASE_DIR.rglob("data_val.hdf5"))

print(f"Found {len(train_files)} train files")
print(f"Found {len(val_files)} val files")

all_inputs = []
all_transcripts = []
all_behavior = []

def process_file(path: Path, split: str):
    global all_inputs, all_transcripts, all_behavior
    session = path.parent.name  # e.g. 't15.2023.08.11'
    print(f"Processing {split} file: {path}")

    with h5py.File(path, "r") as f:
        for trial_name in f.keys():
            g = f[trial_name]

            # neural features [T, C]
            feats = np.array(g["input_features"], dtype=np.float32)  # (T, C)

            # transcription: integer codes (length 500, with zero padding)
            trans = np.array(g["transcription"], dtype=np.int64).reshape(-1)
            nz = np.nonzero(trans)[0]
            if nz.size == 0:
                # completely empty transcript, skip
                continue
            trans = trans[: nz[-1] + 1]  # trim trailing zeros

            all_inputs.append(feats)
            all_transcripts.append(trans)
            all_behavior.append(
                {
                    "session": session,
                    "split": split,
                    "trial": trial_name,
                    "input_len": int(feats.shape[0]),
                    "n_channels": int(feats.shape[1]),
                    "transcript_len": int(trans.shape[0]),
                }
            )

# process all train + val (test has no labels, so skip for now)
for p in train_files:
    process_file(p, "train")

for p in val_files:
    process_file(p, "val")

print("\nTotal trials collected:", len(all_inputs))

if len(all_inputs) == 0:
    print("⚠️ No data collected – something is still wrong with paths.")
    raise SystemExit

# ====== SAVE NPY FILES ======
decoder_inputs = np.array(all_inputs, dtype=object)
transcripts = np.array(all_transcripts, dtype=object)
behavior = np.array(all_behavior, dtype=object)

np.save(OUT_DIR / "decoder_inputs.npy", decoder_inputs)
np.save(OUT_DIR / "transcripts.npy", transcripts)
np.save(OUT_DIR / "behavior.npy", behavior)

meta = {
    "num_trials": len(all_inputs),
    "base_dir": str(BASE_DIR),
    "description": "Converted from brain-to-text-25 hdf5_data_final (train+val)",
}
with open(OUT_DIR / "task_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved:")
print("  decoder_inputs.npy:", decoder_inputs.shape)
print("  transcripts.npy   :", transcripts.shape)
print("  behavior.npy      :", behavior.shape)
print("  task_metadata.json")
