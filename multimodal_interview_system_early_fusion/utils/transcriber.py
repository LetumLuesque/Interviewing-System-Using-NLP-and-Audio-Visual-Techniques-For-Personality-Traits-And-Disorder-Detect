"""
AudioTranscriber — from-scratch Conformer-CTC ASR backend.

Drop-in replacement for the previous Whisper-based transcriber.
Same class name, same .transcribe() interface — no other files need changes.

Supports English and Arabic (bilingual character-level vocabulary).

Checkpoint path (after training):
    utils/asr/checkpoints/best_model.pt

To train the model run:
    python utils/asr/train_asr.py
"""
import os
import logging
import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16000

# Path to the trained checkpoint (relative to this file)
_DEFAULT_CKPT = os.path.join(
    os.path.dirname(__file__), "asr", "checkpoints", "best_model.pt"
)


class AudioTranscriber:
    """
    Automatic Speech Recognition using a from-scratch Conformer-CTC model.

    Supports:
        - English  (trained on LibriSpeech train-clean-100)
        - Arabic   (trained on Common Voice Arabic v24.0)

    Usage:
        transcriber = AudioTranscriber()
        text = transcriber.transcribe("path/to/audio.wav")
    """

    model_name = "ConformerCTC-from-scratch"   # kept for logging compatibility

    def __init__(
        self,
        checkpoint_path: str = _DEFAULT_CKPT,
        device: str = None,
        use_beam: bool = False,
        beam_width: int = 10,
    ):
        self.logger = logging.getLogger(__name__)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_beam   = use_beam
        self.beam_width = beam_width
        self.model = None

        self._load_model(checkpoint_path)

    # ------------------------------------------------------------------ #

    def _load_model(self, checkpoint_path: str) -> None:
        """Load the ConformerCTC model from a checkpoint."""
        try:
            from utils.asr.asr_model import ConformerCTCModel
            if os.path.exists(checkpoint_path):
                self.logger.info(f"Loading ConformerCTC from {checkpoint_path}")
                self.model, _ = ConformerCTCModel.load(checkpoint_path, device=self.device)
                self.model.eval()
                self.logger.info("ConformerCTC model loaded successfully.")
            else:
                self.logger.warning(
                    f"Checkpoint not found at {checkpoint_path}. "
                    "Run `python utils/asr/train_asr.py` to train the model first."
                )
        except Exception as e:
            self.logger.error(f"Failed to load ConformerCTC model: {e}")
            self.model = None

    # ------------------------------------------------------------------ #

    def transcribe(self, audio_input) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_input : str  — path to audio/video file (wav, mp3, mp4, …)
                          OR
                          np.ndarray — raw waveform at 16 kHz

        Returns:
            Transcribed text string (empty string on failure).
        """
        if self.model is None:
            self.logger.warning(
                "ASR model not initialised — returning empty string. "
                "Train the model first: python utils/asr/train_asr.py"
            )
            return ""

        try:
            self.logger.info("Starting transcription (ConformerCTC)...")

            # ---- load waveform ---------------------------------------- #
            if isinstance(audio_input, str):
                waveform, sr = torchaudio.load(audio_input)
                waveform = waveform.mean(dim=0)          # stereo → mono
                if sr != SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

            elif isinstance(audio_input, np.ndarray):
                waveform = torch.from_numpy(audio_input.astype(np.float32))
                if waveform.dim() == 2:
                    waveform = waveform.mean(dim=0)

            else:
                self.logger.error(f"Unsupported audio_input type: {type(audio_input)}")
                return ""

            # ---- transcribe ------------------------------------------- #
            text = self.model.transcribe(
                waveform,
                use_beam=self.use_beam,
                beam_width=self.beam_width,
            )
            self.logger.info(f"Transcription complete. Length: {len(text)} chars")
            return text

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return ""
