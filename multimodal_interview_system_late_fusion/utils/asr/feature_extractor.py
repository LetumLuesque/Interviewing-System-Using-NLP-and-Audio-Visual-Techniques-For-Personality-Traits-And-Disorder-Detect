"""
From-scratch mel-spectrogram feature extractor with SpecAugment.
Pure PyTorch / torchaudio — no librosa dependency for ASR.
"""
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from typing import Optional


class MelSpectrogramExtractor(nn.Module):
    """
    Converts raw waveform → log mel-spectrogram features.

    Output shape: [B, T, n_mels]  (batch, time-frames, mel-bins)

    SpecAugment (applied during training only):
        - Time masking:  masks up to `time_mask_param` consecutive time steps
        - Freq masking:  masks up to `freq_mask_param` consecutive mel bins
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,           # 25ms window @ 16kHz
        hop_length: int = 160,      # 10ms hop   @ 16kHz
        f_min: float = 0.0,
        f_max: Optional[float] = 8000.0,
        # SpecAugment params
        time_mask_param: int = 50,
        freq_mask_param: int = 15,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ):
        super().__init__()

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            window_fn=torch.hann_window,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        # SpecAugment transforms (applied independently per call)
        self.time_masking = nn.ModuleList(
            [T.TimeMasking(time_mask_param=time_mask_param) for _ in range(num_time_masks)]
        )
        self.freq_masking = nn.ModuleList(
            [T.FrequencyMasking(freq_mask_param=freq_mask_param) for _ in range(num_freq_masks)]
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, samples] or [samples]  (float32, values in [-1, 1])

        Returns:
            features: [B, T, n_mels]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)   # [1, samples]

        # [B, n_mels, T]
        mel = self.mel_transform(waveform)
        mel = self.amplitude_to_db(mel)

        # SpecAugment (only during training)
        if self.training:
            for tm in self.time_masking:
                mel = tm(mel)
            for fm in self.freq_masking:
                mel = fm(mel)

        # Normalize: zero mean, unit variance across time+freq
        mean = mel.mean(dim=(-2, -1), keepdim=True)
        std  = mel.std(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
        mel  = (mel - mean) / std

        # [B, T, n_mels]
        return mel.transpose(-2, -1)
