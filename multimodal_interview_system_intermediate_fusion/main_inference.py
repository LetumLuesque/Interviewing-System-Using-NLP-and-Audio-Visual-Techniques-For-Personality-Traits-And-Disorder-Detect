"""
Main inference script for Multimodal Interview Analysis System
"""
import argparse
import yaml
import torch
from pathlib import Path

from utils.helpers import load_config, get_device
from utils.logger import setup_logger, load_checkpoint
from data.processors.video_processor import VideoProcessor
from data.processors.audio_processor import AudioProcessor
from data.processors.text_processor import TextProcessor
from models.multimodal_model import MultimodalInterviewModel
from inference.pipeline import InferencePipeline
from inference.report_generator import HRReportGenerator


def main():
    parser = argparse.ArgumentParser(description='Run Inference on Interview Video')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--audio', type=str, default=None,
                       help='Path to audio file (optional if audio is in video)')
    parser.add_argument('--transcript', type=str, default=None,
                       help='Path to transcript file or transcript text')
    parser.add_argument('--output_dir', type=str, default='reports',
                       help='Output directory for reports')
    parser.add_argument('--candidate_name', type=str, default=None,
                       help='Candidate name for report')
    parser.add_argument('--position', type=str, default=None,
                       help='Position applied for')
    parser.add_argument('--cultural_group', type=str, default=None,
                       help='Cultural group label')
    parser.add_argument('--show_3d', action='store_true',
                       help='Launch 3D personality visualization')
    parser.add_argument('--viz3d_path', type=str, default=None,
                       help='Path to PersonalityViz3D.exe')
    parser.add_argument('--fusion_method', type=str, default=None,
                       help='Override fusion/aggregation method (concat, mean, weighted)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('inference')
    
    logger.info("=" * 80)
    logger.info("Multimodal Interview Analysis System - Inference")
    logger.info("=" * 80)
    
    # Load configuration
    model_config = load_config(args.config)
    
    # Override fusion method if specified via CLI
    if args.fusion_method:
        fm = args.fusion_method.lower()
        logger.info(f"Overriding fusion/aggregation method to: {fm}")
        if 'fusion' not in model_config:
            model_config['fusion'] = {}
        if 'intermediate_fusion' not in model_config['fusion']:
            model_config['fusion']['intermediate_fusion'] = {}
        model_config['fusion']['intermediate_fusion']['aggregation'] = fm
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Initialize processors
    data_config = model_config.get('data', {})
    video_config = data_config.get('video', {})
    audio_config = data_config.get('audio', {})
    text_config = data_config.get('text', {})
    
    video_processor = VideoProcessor(
        fps=video_config.get('fps', 1),
        frame_size=video_config.get('frame_size', 224),
        use_face_detection=True
    )
    
    audio_processor = AudioProcessor(
        sample_rate=audio_config.get('sample_rate', 16000),
        mfcc_coefficients=audio_config.get('mfcc_coefficients', 40),
        hop_length=audio_config.get('hop_length', 160)
    )
    
    text_processor = TextProcessor(
        tokenizer_name=text_config.get('tokenizer', 'bert-base-uncased'),
        max_length=text_config.get('max_length', 512)
    )
    
    # Initialize model
    model = MultimodalInterviewModel(model_config)
    model.to(device)
    
    # Load checkpoint
    load_checkpoint(args.checkpoint, model)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    
    # Create inference pipeline
    pipeline = InferencePipeline(
        model=model,
        video_processor=video_processor,
        audio_processor=audio_processor,
        text_processor=text_processor,
        device=device
    )
    
    # Process interview
    logger.info(f"Processing interview: {args.video}")
    results = pipeline.process_interview(
        video_path=args.video,
        audio_path=args.audio,
        transcript=args.transcript,
        cultural_group=args.cultural_group
    )
    
    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("Analysis Results")
    logger.info("=" * 80)
    
    trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    traits = results['traits']
    
    logger.info("\nBig Five Personality Traits:")
    for name, score in zip(trait_names, traits):
        logger.info(f"  {name}: {score:.2f}")
    
    logger.info(f"\nDeception Probability: {results['deception_probability']:.4f}")
    
    if results.get('confidence') is not None:
        logger.info("\nConfidence Scores:")
        for name, conf in zip(trait_names, results['confidence']):
            logger.info(f"  {name}: {conf:.4f}")
    
    # Generate HR report
    logger.info("\nGenerating HR report...")
    report_generator = HRReportGenerator(
        output_dir=args.output_dir,
        viz3d_exe_path=args.viz3d_path
    )
    report_path = report_generator.generate_report(
        analysis_results=results,
        candidate_name=args.candidate_name,
        position=args.position,
        save_html=True,
        show_3d=args.show_3d
    )
    
    # Save results as JSON for the GUI app
    import json
    json_path = Path(args.output_dir) / 'analysis_results.json'
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (torch.Tensor,)):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj

    # Create a serializable dictionary
    json_results = {
        'traits': [convert_numpy(x) for x in results.get('traits', [])],
        'deception_probability': convert_numpy(results.get('deception_probability', 0.0)),
        'confidence': [convert_numpy(x) for x in results.get('confidence', [])] if results.get('confidence') is not None else None,
        'report_path': str(report_path)
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=4)
        
    logger.info(f"Results saved to JSON: {json_path}")
    
    logger.info(f"\nReport saved to: {report_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

















