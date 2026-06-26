"""
Training script for the from-scratch Conformer-CTC ASR model.

Datasets:
  - LibriSpeech train-clean-100 (English, auto-downloaded)
  - Common Voice Arabic v24.0   (Arabic, manual download)
      expected at: D:/Grad/Datasets/cv-corpus-24.0-2025-12-05/ar/

Usage:
    python utils/asr/train_asr.py

Checkpoints saved to:  utils/asr/checkpoints/
Best model saved as:   utils/asr/checkpoints/best_model.pt
"""

import os
import sys
import math
import time
import logging
import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Optional

# Make sibling modules importable when run as a script
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio
import torchaudio.transforms as T

from asr_model import ConformerCTCModel
from tokenizer import CharTokenizer

# ============================================================================ #
# Config                                                                        #
# ============================================================================ #

# LIBRISPEECH_ROOT is the parent folder where 'LibriSpeech' subfolder will be (created by torchaudio)
LIBRISPEECH_ROOT = "D:/Grad/Datasets"
CV_ARABIC_ROOT   = "D:/Grad/Datasets/cv-corpus-24.0-2025-12-05/ar"

CHECKPOINT_DIR   = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_FILE         = os.path.join(CHECKPOINT_DIR, "train_asr.log")

# Model hyper-parameters (fits in 12 GB VRAM)
MODEL_CFG = dict(
    n_mels      = 80,
    d_model     = 256,
    num_layers  = 12,
    num_heads   = 4,
    kernel_size = 31,
    ff_expansion= 4,
    dropout     = 0.1,
)

# Training hyper-parameters
BATCH_SIZE     = 8           # per-GPU batch
GRAD_ACCUM     = 4           # effective batch = 32
EPOCHS         = 50
LR             = 1e-4
WARMUP_STEPS   = 1000
MAX_AUDIO_SEC  = 20.0        # skip clips longer than this
SAMPLE_RATE    = 16000
NUM_WORKERS    = 4
SEED           = 42

# ============================================================================ #
# Logging                                                                       #
# ============================================================================ #

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================ #
# Datasets                                                                      #
# ============================================================================ #

class LibriSpeechDataset(Dataset):
    """Manual LibriSpeech loader using librosa and pathlib."""

    def __init__(self, root: str, split: str = "train-clean-100", tokenizer: CharTokenizer = None):
        self.tokenizer = tokenizer or CharTokenizer()
        self.root = Path(root) / "LibriSpeech" / split
        logger.info(f"Loading LibriSpeech {split} from {self.root} ...")

        if not self.root.exists():
            # Try to trigger download if parent exists but subfolder doesn't
            parent = Path(root)
            logger.info(f"Subfolder not found, trying torchaudio download to {parent}...")
            # We call this just to ensure it downloads if missing, 
            # though it might fail to load later, we just need the files.
            try:
                torchaudio.datasets.LIBRISPEECH(root=root, url=split, download=True)
            except Exception as e:
                logger.info(f"Download check finished/skipped: {e}")

        self.samples = []
        # Structure: root/reader/chapter/*.flac
        for trans_file in self.root.rglob("*.trans.txt"):
            parent_dir = trans_file.parent
            with open(trans_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    fileid = parts[0]
                    transcript = " ".join(parts[1:]).lower()
                    audio_path = parent_dir / f"{fileid}.flac"
                    if audio_path.exists():
                        self.samples.append((str(audio_path), transcript))

        if not self.samples:
            logger.error(f"No samples found in {self.root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        audio_path, transcript = self.samples[idx]
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            waveform = torch.from_numpy(audio)
        except Exception as e:
            logger.warning(f"Failed to load {audio_path}: {e}")
            return torch.zeros(SAMPLE_RATE), []

        label_ids = self.tokenizer.encode(transcript)
        return waveform, label_ids


class CommonVoiceArabicDataset(Dataset):
    """
    Reads Common Voice Arabic v24.0 from the extracted folder.
    Expected structure:
        <root>/
            clips/   *.mp3
            train.tsv
            dev.tsv
            test.tsv
    """

    def __init__(self, root: str, split: str = "train", tokenizer: CharTokenizer = None):
        self.tokenizer = tokenizer or CharTokenizer()
        self.clips_dir = os.path.join(root, "clips")
        tsv_path       = os.path.join(root, f"{split}.tsv")

        logger.info(f"Loading Common Voice Arabic {split} from {tsv_path} ...")

        self.samples: List[Tuple[str, str]] = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                mp3_path  = os.path.join(self.clips_dir, row["path"])
                sentence  = row["sentence"].strip()
                if os.path.exists(mp3_path) and sentence:
                    self.samples.append((mp3_path, sentence))

        logger.info(f"CommonVoice Arabic {split}: {len(self.samples)} samples loaded")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        mp3_path, sentence = self.samples[idx]
        try:
            import librosa
            audio, sr = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
            waveform = torch.from_numpy(audio)
        except Exception as e:
            logger.warning(f"Failed to load {mp3_path}: {e}")
            return torch.zeros(SAMPLE_RATE), []

        label_ids = self.tokenizer.encode(sentence)
        return waveform, label_ids


# ============================================================================ #
# Collate                                                                       #
# ============================================================================ #

def collate_fn(batch):
    """
    Pads waveforms + label sequences for batching.
    Filters out samples that are too long.
    """
    # Filter by max audio length
    filtered = [
        (w, l) for w, l in batch
        if len(w) <= MAX_AUDIO_SEC * SAMPLE_RATE and len(l) > 0
    ]
    if not filtered:
        return None

    waveforms, labels = zip(*filtered)
    wav_lengths = torch.tensor([len(w) for w in waveforms])
    lbl_lengths = torch.tensor([len(l) for l in labels])

    # Pad waveforms
    max_wav = max(len(w) for w in waveforms)
    padded_wavs = torch.zeros(len(waveforms), max_wav)
    for i, w in enumerate(waveforms):
        padded_wavs[i, :len(w)] = w if isinstance(w, torch.Tensor) else torch.tensor(w)

    # Pad labels (pad_id = 1)
    max_lbl = max(len(l) for l in labels)
    padded_lbls = torch.ones(len(labels), max_lbl, dtype=torch.long)
    for i, l in enumerate(labels):
        padded_lbls[i, :len(l)] = torch.tensor(l, dtype=torch.long)

    return padded_wavs, wav_lengths, padded_lbls, lbl_lengths


# ============================================================================ #
# WER                                                                           #
# ============================================================================ #

def _edit_distance(ref: List[str], hyp: List[str]) -> int:
    """Standard edit distance (for WER computation)."""
    R, H = len(ref), len(hyp)
    dp = list(range(H + 1))
    for i in range(1, R + 1):
        new_dp = [i] + [0] * H
        for j in range(1, H + 1):
            if ref[i-1] == hyp[j-1]:
                new_dp[j] = dp[j-1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j-1], dp[j-1])
        dp = new_dp
    return dp[H]


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    total_words = 0
    total_edits = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_words += max(len(ref_words), 1)
        total_edits += _edit_distance(ref_words, hyp_words)
    return total_edits / max(total_words, 1)


# ============================================================================ #
# Training                                                                      #
# ============================================================================ #

def get_lr(step: int, d_model: int, warmup: int) -> float:
    """Transformer / Conformer learning-rate schedule."""
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def train():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    tokenizer = CharTokenizer()
    tokenizer.save(os.path.join(CHECKPOINT_DIR, "vocab.json"))

    # -------- datasets -------- #
    datasets = []

    # LibriSpeech English
    try:
        ls_train = LibriSpeechDataset(LIBRISPEECH_ROOT, "train-clean-100", tokenizer)
        datasets.append(ls_train)
        logger.info(f"LibriSpeech train-clean-100: {len(ls_train)} samples")
    except Exception as e:
        logger.warning(f"LibriSpeech not available: {e}")

    # Common Voice Arabic
    if os.path.exists(CV_ARABIC_ROOT):
        try:
            cv_train = CommonVoiceArabicDataset(CV_ARABIC_ROOT, "train", tokenizer)
            datasets.append(cv_train)
            logger.info(f"Common Voice Arabic train: {len(cv_train)} samples")
        except Exception as e:
            logger.warning(f"Common Voice Arabic load failed: {e}")
    else:
        logger.warning(f"Common Voice Arabic not found at {CV_ARABIC_ROOT}")

    if len(datasets) < 2:
        logger.error("Missing required datasets for bilingual training.")
        logger.error(f"Datasets loaded: {[f'{type(d).__name__}' for d in datasets]}")
        raise RuntimeError("Both English (LibriSpeech) and Arabic (Common Voice) are required.")

    train_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    train_loader  = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Optional: Common Voice Arabic dev set for validation
    val_loader = None
    if os.path.exists(CV_ARABIC_ROOT):
        try:
            cv_dev = CommonVoiceArabicDataset(CV_ARABIC_ROOT, "dev", tokenizer)
            val_loader = DataLoader(cv_dev, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, collate_fn=collate_fn)
            logger.info(f"Validation: Common Voice Arabic dev ({len(cv_dev)} samples)")
        except Exception as e:
            logger.warning(f"Could not load CV dev set: {e}")

    # -------- model -------- #
    model = ConformerCTCModel(**MODEL_CFG, tokenizer=tokenizer).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    ctc_loss = nn.CTCLoss(blank=tokenizer.BLANK_ID, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_wer  = float("inf")
    global_step = 0
    start_epoch = 1

    # -------- resume logic -------- #
    import glob
    ckpt_files = glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt"))
    if ckpt_files:
        # Sort by epoch number: epoch_005.pt -> 5
        ckpt_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
        latest_ckpt = ckpt_files[-1]
        logger.info(f"Resuming from checkpoint: {latest_ckpt}")
        
        # We need to initialize optimizer/scheduler before loading their state
        model.to(device) # ensure model on device first
        _, start_epoch = model.load(latest_ckpt, device=device, optimizer=optimizer, scheduler=scheduler)
        start_epoch += 1 # start from next epoch
        logger.info(f"Starting from epoch {start_epoch}")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss   = 0.0
        epoch_steps  = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            wavs, wav_lens, lbls, lbl_lens = batch
            wavs = wavs.to(device)
            wav_lens = wav_lens.to(device)
            lbls = lbls.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                log_probs, out_lens = model(wavs, wav_lens)
                # CTC expects [T, B, C]
                log_probs_ctc = log_probs.permute(1, 0, 2)
                if out_lens is None:
                    out_lens = torch.full((wavs.size(0),), log_probs.size(1), dtype=torch.long, device=device)
                loss = ctc_loss(log_probs_ctc, lbls, out_lens, lbl_lens)
                loss = loss / GRAD_ACCUM

            if not torch.isfinite(loss):
                logger.warning(f"Loss is {loss.item()} at Epoch {epoch} Step {batch_idx}. Skipping batch.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss  += loss.item() * GRAD_ACCUM
            epoch_steps += 1

            if batch_idx % 200 == 0:
                logger.info(
                    f"Epoch {epoch}/{EPOCHS} | Step {batch_idx}/{len(train_loader)} "
                    f"| Loss {epoch_loss/max(epoch_steps,1):.4f}"
                )

        scheduler.step()
        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"=== Epoch {epoch} done | Avg Loss: {avg_loss:.4f} ===")

        # --------- Validation --------- #
        if val_loader is not None:
            model.eval()
            refs_all, hyps_all = [], []
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    wavs, wav_lens, lbls, lbl_lens = batch
                    wavs = wavs.to(device)
                    log_probs, out_lens = model(wavs, wav_lens)
                    if out_lens is None:
                        out_lens = torch.full((wavs.size(0),), log_probs.size(1), dtype=torch.long)

                    from ctc_decoder import greedy_decode
                    decoded = greedy_decode(log_probs.cpu(), out_lens.cpu(), tokenizer.BLANK_ID)

                    for i, ids in enumerate(decoded):
                        hyp = tokenizer.decode(ids)
                        ref_ids = lbls[i][:lbl_lens[i]].tolist()
                        ref = tokenizer.decode(ref_ids, remove_special=False)
                        hyps_all.append(hyp)
                        refs_all.append(ref)

            wer = compute_wer(refs_all, hyps_all)
            logger.info(f"Epoch {epoch} | Validation WER: {wer*100:.2f}%")

            if wer < best_wer:
                best_wer = wer
                best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
                model.save(best_path)
                logger.info(f"  >> New best WER {best_wer*100:.2f}% — saved to {best_path}")

        # Save latest checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt")
            model.save(ckpt_path, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
            logger.info(f"Checkpoint saved: {ckpt_path}")

    logger.info(f"Training complete. Best WER: {best_wer*100:.2f}%")
    logger.info(f"Best model: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")


# ============================================================================ #

if __name__ == "__main__":
    train()
