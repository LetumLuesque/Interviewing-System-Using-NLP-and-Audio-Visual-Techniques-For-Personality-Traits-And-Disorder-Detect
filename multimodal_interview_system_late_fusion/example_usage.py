"""
Example usage script for Multimodal Interview Analysis System
"""
import torch
from utils.helpers import load_config, get_device
from data.processors.video_processor import VideoProcessor
from data.processors.audio_processor import AudioProcessor
from data.processors.text_processor import TextProcessor
from models.multimodal_model import MultimodalInterviewModel
from inference.pipeline import InferencePipeline
from inference.report_generator import HRReportGenerator


def example_inference():
    """
    Example of running inference on an interview video
    """
    print("=" * 80)
    print("Multimodal Interview Analysis System - Example Usage")
    print("=" * 80)
    
    # Load configuration
    model_config = load_config('config/model_config.yaml')
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize processors
    video_processor = VideoProcessor()
    audio_processor = AudioProcessor()
    text_processor = TextProcessor()
    
    # Initialize model
    model = MultimodalInterviewModel(model_config)
    model.to(device)
    
    # Create inference pipeline
    pipeline = InferencePipeline(
        model=model,
        video_processor=video_processor,
        audio_processor=audio_processor,
        text_processor=text_processor,
        device=device
    )
    
    # Example: Process an interview
    # Replace these paths with your actual data paths
    video_path = "path/to/interview_video.mp4"
    audio_path = "path/to/interview_audio.wav"  # Optional if audio is in video
    transcript = "This is a sample interview transcript..."  # Or path to file
    
    print("\nProcessing interview...")
    results = pipeline.process_interview(
        video_path=video_path,
        audio_path=audio_path,
        transcript=transcript,
        cultural_group="group1"
    )
    
    # Display results
    print("\nResults:")
    trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    for name, score in zip(trait_names, results['traits']):
        print(f"  {name}: {score:.2f}")
    
    print(f"\nDeception Probability: {results['deception_probability']:.4f}")
    
    # Generate report
    print("\nGenerating HR report...")
    report_generator = HRReportGenerator(output_dir="reports")
    report_path = report_generator.generate_report(
        analysis_results=results,
        candidate_name="Example Candidate",
        position="Software Engineer"
    )
    
    print(f"\nReport saved to: {report_path}")
    print("=" * 80)


def example_training_setup():
    """
    Example of setting up training
    """
    print("=" * 80)
    print("Training Setup Example")
    print("=" * 80)
    
    # Load configurations
    train_config = load_config('config/training_config.yaml')
    model_config = load_config('config/model_config.yaml')
    
    # Initialize model
    model = MultimodalInterviewModel(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nTo train the model, run:")
    print("python main_train.py --data_dir data/mdpe_dataset")
    print("=" * 80)


if __name__ == '__main__':
    # Uncomment the example you want to run
    # example_inference()
    example_training_setup()
































