"""
Full ConformerCTC ASR Model — from scratch.

Architecture:
    MelSpectrogramExtractor → ConformerEncoder → Linear CTC head
    → log-softmax → greedy/beam decoder → text

This is a drop-in backend for utils/transcriber.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

try:
    from .feature_extractor import MelSpectrogramExtractor
    from .conformer import ConformerEncoder
    from .tokenizer import CharTokenizer
    from .ctc_decoder import greedy_decode, beam_search_decode
except ImportError:
    from feature_extractor import MelSpectrogramExtractor
    from conformer import ConformerEncoder
    from tokenizer import CharTokenizer
    from ctc_decoder import greedy_decode, beam_search_decode


class ConformerCTCModel(nn.Module):
    """
    End-to-end Conformer-CTC ASR model.

    Parameters
    ----------
    n_mels        : mel bins (default 80)
    d_model       : encoder hidden dim (default 256)
    num_layers    : number of Conformer blocks (default 12)
    num_heads     : attention heads (default 4)
    kernel_size   : depthwise conv kernel (default 31)
    ff_expansion  : FFN expansion factor (default 4)
    dropout       : dropout throughout (default 0.1)
    tokenizer     : CharTokenizer instance (built from vocab in checkpoint,
                    or default if None)
    """

    def __init__(
        self,
        n_mels:      int = 80,
        d_model:     int = 256,
        num_layers:  int = 12,
        num_heads:   int = 4,
        kernel_size: int = 31,
        ff_expansion: int = 4,
        dropout:     float = 0.1,
        tokenizer:   Optional[CharTokenizer] = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer or CharTokenizer()
        vocab_size = self.tokenizer.vocab_size

        # Feature extraction (from scratch — no librosa)
        self.feature_extractor = MelSpectrogramExtractor(n_mels=n_mels)

        # Conformer encoder
        self.encoder = ConformerEncoder(
            n_mels=n_mels,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            kernel_size=kernel_size,
            ff_expansion=ff_expansion,
            dropout=dropout,
        )

        # CTC projection head
        self.ctc_head = nn.Linear(d_model, vocab_size)

    # ----------------------------------------------------------------------- #

    def forward(
        self,
        waveform: torch.Tensor,
        lengths:  Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            waveform : [B, samples]  raw audio at 16 kHz
            lengths  : [B] number of valid samples per batch item (optional)

        Returns:
            log_probs      : [B, T, vocab_size]  (log-softmax)
            output_lengths : [B] valid time steps after subsampling, or None
        """
        # 1. Feature extraction
        features = self.feature_extractor(waveform)    # [B, T, n_mels]

        # Compute frame lengths from sample lengths
        frame_lengths = None
        if lengths is not None:
            hop = 160   # 10ms hop
            frame_lengths = (lengths // hop).clamp(min=1)

        # 2. Conformer encoder
        encoded, output_lengths = self.encoder(features, frame_lengths)  # [B, T', d_model]

        # Always return a valid lengths tensor (T' for every sample in batch)
        if output_lengths is None:
            B, T_prime, _ = encoded.shape
            output_lengths = torch.full((B,), T_prime, dtype=torch.long, device=encoded.device)

        # 3. CTC head
        logits   = self.ctc_head(encoded)             # [B, T', vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, output_lengths


    # ----------------------------------------------------------------------- #

    @torch.no_grad()
    def transcribe(
        self,
        waveform:   torch.Tensor,
        use_beam:   bool = False,
        beam_width: int = 10,
    ) -> str:
        """
        End-to-end: waveform → transcript string.

        Args:
            waveform   : [samples] or [1, samples]  at 16 kHz
            use_beam   : use beam search instead of greedy
            beam_width : beam width (only if use_beam=True)

        Returns:
            Decoded transcript string.
        """
        self.eval()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)   # [1, samples]

        device = next(self.parameters()).device
        waveform = waveform.to(device)

        log_probs, output_lengths = self.forward(waveform)

        # Build a lengths tensor for the decoder
        if output_lengths is None:
            T = log_probs.size(1)
            output_lengths = torch.tensor([T], device="cpu")
        else:
            output_lengths = output_lengths.cpu()

        log_probs_cpu = log_probs.cpu()

        if use_beam:
            token_ids = beam_search_decode(log_probs_cpu, output_lengths,
                                           blank_id=self.tokenizer.BLANK_ID,
                                           beam_width=beam_width)[0]
        else:
            token_ids = greedy_decode(log_probs_cpu, output_lengths,
                                      blank_id=self.tokenizer.BLANK_ID)[0]

        return self.tokenizer.decode(token_ids, remove_special=True)

    # ----------------------------------------------------------------------- #

    def save(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        epoch:     int = 0,
    ) -> None:
        """Save model weights + tokenizer vocab + training state to a checkpoint."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "model_state": self.state_dict(),
            "vocab":       self.tokenizer.vocab,
            "epoch":       epoch,
            "config": {
                "n_mels":      self.feature_extractor.mel_transform.n_mels,
                "d_model":     self.encoder.norm.normalized_shape[0],
                "num_layers":  len(self.encoder.layers),
                "num_heads":   self.encoder.layers[0].mhsa.attn.num_heads,
                "kernel_size": self.encoder.layers[0].conv.dw.kernel_size[0],
            }
        }
        if optimizer is not None:
            state["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state"] = scheduler.state_dict()

        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> tuple["ConformerCTCModel", int]:
        """
        Load a saved checkpoint. Returns (model, epoch).
        If optimizer/scheduler are provided, their states are also loaded.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = ckpt.get("config", {})
        tok  = CharTokenizer(vocab=ckpt["vocab"])

        model = cls(
            n_mels=cfg.get("n_mels", 80),
            d_model=cfg.get("d_model", 256),
            num_layers=cfg.get("num_layers", 12),
            num_heads=cfg.get("num_heads", 4),
            kernel_size=cfg.get("kernel_size", 31),
            tokenizer=tok,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])

        return model, ckpt.get("epoch", 0)

