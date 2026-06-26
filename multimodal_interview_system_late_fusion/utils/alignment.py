"""
Temporal alignment utilities for multimodal data
"""
import numpy as np
import torch


def align_modalities(video_frames, audio_features, text_tokens, video_fps=1, audio_fps=10):
    """
    Align video, audio, and text features temporally
    
    Args:
        video_frames: Video frames tensor [B, T_v, C, H, W]
        audio_features: Audio features tensor [B, T_a, F]
        text_tokens: Text tokens tensor [B, T_t]
        video_fps: Frames per second for video
        audio_fps: Frames per second for audio
        
    Returns:
        Aligned features with same temporal dimension
    """
    B, T_v, C, H, W = video_frames.shape
    _, T_a, F = audio_features.shape
    
    # Target temporal dimension (use video as reference)
    target_length = T_v
    
    # Interpolate audio to match video temporal resolution
    if T_a != target_length:
        audio_features = torch.nn.functional.interpolate(
            audio_features.transpose(1, 2),  # [B, F, T_a]
            size=target_length,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # [B, T_v, F]
    
    # For text, we repeat tokens across temporal dimension
    # or use attention-based alignment
    if text_tokens is not None:
        _, T_t = text_tokens.shape
        if T_t != target_length:
            # Simple repetition (can be improved with attention)
            repeat_factor = max(1, target_length // T_t)
            remainder = target_length % T_t
            
            if T_t > 0:
                text_aligned = text_tokens.repeat(1, repeat_factor)
                if remainder > 0:
                    text_aligned = torch.cat([
                        text_aligned,
                        text_tokens[:, :remainder]
                    ], dim=1)
                text_tokens = text_aligned[:, :target_length]
    
    return video_frames, audio_features, text_tokens


def temporal_padding(sequence, target_length, pad_value=0):
    """
    Pad or truncate sequence to target length
    
    Args:
        sequence: Input sequence tensor
        target_length: Desired length
        pad_value: Value to use for padding
        
    Returns:
        Padded/truncated sequence
    """
    current_length = sequence.shape[1]
    
    if current_length < target_length:
        # Pad
        pad_size = target_length - current_length
        padding = torch.full(
            (sequence.shape[0], pad_size, *sequence.shape[2:]),
            pad_value,
            dtype=sequence.dtype,
            device=sequence.device
        )
        sequence = torch.cat([sequence, padding], dim=1)
    elif current_length > target_length:
        # Truncate
        sequence = sequence[:, :target_length]
    
    return sequence


def synchronize_timestamps(video_timestamps, audio_timestamps, text_timestamps):
    """
    Synchronize timestamps across modalities
    
    Args:
        video_timestamps: List of video frame timestamps
        audio_timestamps: List of audio frame timestamps
        text_timestamps: List of text segment timestamps
        
    Returns:
        Synchronized indices for each modality
    """
    # Find common time range
    start_time = max(
        min(video_timestamps),
        min(audio_timestamps),
        min(text_timestamps)
    )
    end_time = min(
        max(video_timestamps),
        max(audio_timestamps),
        max(text_timestamps)
    )
    
    # Find indices within common range
    video_indices = [
        i for i, t in enumerate(video_timestamps)
        if start_time <= t <= end_time
    ]
    audio_indices = [
        i for i, t in enumerate(audio_timestamps)
        if start_time <= t <= end_time
    ]
    text_indices = [
        i for i, t in enumerate(text_timestamps)
        if start_time <= t <= end_time
    ]
    
    return video_indices, audio_indices, text_indices
































