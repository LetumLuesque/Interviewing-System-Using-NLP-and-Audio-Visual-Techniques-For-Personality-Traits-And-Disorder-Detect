"""
Inference pipeline for processing interview videos
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import os

from data.processors.video_processor import VideoProcessor
from data.processors.audio_processor import AudioProcessor
from data.processors.text_processor import TextProcessor
from utils.helpers import get_device
from utils.logger import load_checkpoint
from utils.alignment import align_modalities
from utils.transcriber import AudioTranscriber


class InferencePipeline:
    """
    End-to-end inference pipeline for interview analysis
    """
    
    def __init__(
        self,
        model: nn.Module,
        video_processor: Optional[VideoProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        text_processor: Optional[TextProcessor] = None,
        device: Optional[torch.device] = None,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize inference pipeline
        
        Args:
            model: Trained model
            video_processor: Video processor instance
            audio_processor: Audio processor instance
            text_processor: Text processor instance
            device: Device to run inference on
            checkpoint_path: Optional path to model checkpoint
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            load_checkpoint(checkpoint_path, self.model)
        
        # Initialize processors
        self.video_processor = video_processor or VideoProcessor()
        self.audio_processor = audio_processor or AudioProcessor()
        self.text_processor = text_processor or TextProcessor()
    
    def process_video_file(
        self,
        video_path: str,
        audio_path: Optional[str] = None,
        transcript_path: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process interview video file
        
        Args:
            video_path: Path to video file
            audio_path: Optional path to audio file (if separate)
            transcript_path: Optional path to transcript file
            
        Returns:
            Model predictions
        """
        # Process video
        video_features = None
        if os.path.exists(video_path):
            video_features = self.video_processor.process_video(video_path)
            video_features = video_features.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Process audio
        audio_features = None
        if audio_path and os.path.exists(audio_path):
            mfcc, _ = self.audio_processor.process_audio(audio_path, pad_to_length=300)
            audio_features = mfcc.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Process text
        text_features = None
        if transcript_path and os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            text_data = self.text_processor.process_transcript(transcript)
            text_features = {
                'input_ids': text_data['token_ids'].to(self.device),
                'attention_mask': text_data['attention_mask'].to(self.device)
            }
        
        # Align modalities if needed
        if video_features is not None and audio_features is not None:
            video_features, audio_features, _ = align_modalities(
                video_features, audio_features, None
            )
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(video_features, audio_features, text_features)
        
        return predictions
    


    def process_interview(
        self,
        video_path: str,
        audio_path: Optional[str] = None,
        transcript: Optional[str] = None,
        cultural_group: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Process complete interview and return analysis
        
        Args:
            video_path: Path to video file
            audio_path: Optional path to audio file
            transcript: Optional transcript text (or path to file)
            cultural_group: Optional cultural group label (e.g., 'en', 'zh', 'English', etc.)
            
        Returns:
            Complete analysis dictionary
        """
        # Handle transcript (string or file path) or Auto-Transcribe
        transcript_path = None
        
        if transcript:
            if os.path.exists(transcript):
                transcript_path = transcript
            else:
                # Treat as text, save to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                    f.write(transcript)
                    transcript_path = f.name
        else:
            # AUTO-TRANSCRIPTION
            try:
                print("No transcript provided. Starting Auto-Transcription...")
                transcriber = AudioTranscriber()
                
                # Use audio path if available, else video path
                source_path = audio_path if (audio_path and os.path.exists(audio_path)) else video_path
                
                if source_path and os.path.exists(source_path):
                    transcribed_text = transcriber.transcribe(source_path)
                    
                    if transcribed_text:
                        print(f"Auto-Transcription successful (Length: {len(transcribed_text)})")
                        # Save to temp file
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                            f.write(transcribed_text)
                            transcript_path = f.name
                    else:
                        print("Warning: Auto-Transcription returned empty text.")
            except Exception as e:
                print(f"Warning: Auto-Transcription failed: {e}")
        
        # Process video
        predictions = self.process_video_file(video_path, audio_path, transcript_path)

        # Prepare cultural group tensor if provided
        cultural_group_tensor = None
        if cultural_group:
            # Map string to ID (0-4)
            # 0: Chinese (zh), 1: English (en), 2: Arabic (ar), 3: Hindi (hi), 4: Spanish (es)
            cg_map = {
                'zh': 0, 'chinese': 0, 
                'en': 1, 'english': 1, 'north-american': 1,
                'ar': 2, 'arabic': 2,
                'hi': 3, 'hindi': 3,
                'es': 4, 'spanish': 4
            }
            cg_id = cg_map.get(cultural_group.lower(), 0) # Default to 0 (Chinese) if unknown
            cultural_group_tensor = torch.tensor([cg_id], dtype=torch.long).to(self.device)

        # Run inference again with cultural group if valid
        # (Optimization: convert process_video_file to just return features, then run model once)
        if cultural_group_tensor:
            # Process video features
            video_features = None
            if os.path.exists(video_path):
                video_features = self.video_processor.process_video(video_path)
                video_features = video_features.unsqueeze(0).to(self.device)
            
            # Process audio features
            audio_features = None
            if audio_path and os.path.exists(audio_path):
                mfcc, _ = self.audio_processor.process_audio(audio_path, pad_to_length=300)
                audio_features = mfcc.unsqueeze(0).to(self.device)
            elif not audio_path and os.path.exists(video_path):
                try:
                    mfcc, _ = self.audio_processor.process_audio(video_path, pad_to_length=300)
                    audio_features = mfcc.unsqueeze(0).to(self.device)
                except:
                    pass

            # Process text features
            text_features = None
            if transcript_path and os.path.exists(transcript_path):
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                text_data = self.text_processor.process_transcript(content)
                text_features = {
                    'input_ids': text_data['token_ids'].to(self.device),
                    'attention_mask': text_data['attention_mask'].to(self.device)
                }

            # Align modalities
            if video_features is not None and audio_features is not None:
                video_features, audio_features, _ = align_modalities(
                    video_features, audio_features, None
                )

            # Run inference WITH culture
            with torch.no_grad():
                predictions = self.model(
                    video_features, 
                    audio_features, 
                    text_features, 
                    cultural_group=cultural_group_tensor
                )
        
        # Extract results
        traits = predictions['traits'].cpu().numpy()[0]  # Remove batch dimension
        deception_prob = predictions['deception_prob'].item()
        confidence = predictions.get('confidence', None)
        if confidence is not None:
            confidence = confidence.cpu().numpy()[0]
        
        # Clean up temp file if created
        if transcript_path and transcript_path != transcript:
            try:
                if os.path.exists(transcript_path):
                    os.unlink(transcript_path)
            except:
                pass
        
        # Get modality contributions (weights)
        modality_weights = self.model.get_modality_contributions()
        
        if isinstance(modality_weights, dict):
            mod_conf = {
                'visual': modality_weights.get('visual', torch.tensor(0.33)).item(),
                'audio': modality_weights.get('audio', torch.tensor(0.33)).item(),
                'text': modality_weights.get('text', torch.tensor(0.33)).item()
            }
        else:
            # Early/other fusion fallback
            mod_conf = {
                'visual': modality_weights[0].item(),
                'audio': modality_weights[1].item(),
                'text': modality_weights[2].item()
            }

        result = {
            'traits': traits,
            'deception_probability': deception_prob,
            'confidence': confidence,
            'modality_confidence': mod_conf,
            'cultural_group': cultural_group
        }
        
        return result
    
    def batch_process(
        self,
        interview_paths: list,
        batch_size: int = 8
    ) -> list:
        """
        Process multiple interviews in batch
        
        Args:
            interview_paths: List of dictionaries with 'video_path', 'audio_path', 'transcript_path'
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(interview_paths), batch_size):
            batch = interview_paths[i:i+batch_size]
            
            batch_results = []
            for interview in batch:
                result = self.process_interview(
                    interview['video_path'],
                    interview.get('audio_path'),
                    interview.get('transcript')
                )
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results




















