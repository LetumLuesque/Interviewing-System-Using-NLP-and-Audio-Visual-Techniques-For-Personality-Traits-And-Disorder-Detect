"""
Accuracy testing module
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List
from training.metrics import MetricsCalculator


class AccuracyTester:
    """
    Test model accuracy on test set
    """
    
    def __init__(self, model, device):
        """
        Initialize accuracy tester
        
        Args:
            model: Trained model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.metrics_calculator = MetricsCalculator()
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Test model on test set
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of test metrics
        """
        self.metrics_calculator.reset()
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                video = batch['video'].to(self.device) if batch['video'] is not None else None
                audio = batch['audio'].to(self.device) if batch['audio'] is not None else None
                text = {k: v.to(self.device) for k, v in batch['text'].items()} if batch['text'] is not None else None
                
                personality_target = batch['personality'].to(self.device)
                deception_target = batch['deception'].to(self.device)
                cultural_groups = batch['cultural_groups']
                
                # Forward pass
                predictions = self.model(video, audio, text)
                
                # Update metrics
                self.metrics_calculator.update(
                    predictions['traits'],
                    personality_target,
                    predictions['deception_logits'],
                    deception_target,
                    cultural_groups
                )
        
        # Compute metrics
        metrics = self.metrics_calculator.compute()
        
        return metrics
    
    def compare_with_baselines(
        self,
        test_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare test results with baseline methods
        
        Args:
            test_metrics: Current model metrics
            baseline_metrics: Baseline model metrics
            
        Returns:
            Comparison dictionary
        """
        comparison = {}
        
        for metric_name in test_metrics:
            if metric_name in baseline_metrics:
                improvement = test_metrics[metric_name] - baseline_metrics[metric_name]
                improvement_pct = (improvement / baseline_metrics[metric_name]) * 100 if baseline_metrics[metric_name] != 0 else 0
                
                comparison[metric_name] = {
                    'baseline': baseline_metrics[metric_name],
                    'current': test_metrics[metric_name],
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }
        
        return comparison




















