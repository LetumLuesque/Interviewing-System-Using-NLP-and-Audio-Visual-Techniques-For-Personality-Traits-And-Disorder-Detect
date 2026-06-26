"""
HR evaluation module for interview analysis
"""
import torch
import numpy as np
from typing import Dict, List, Optional
from training.metrics import MetricsCalculator


class HREvaluator:
    """
    HR-specific evaluation metrics and analysis
    """
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_interview(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        cultural_group: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single interview
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            cultural_group: Cultural group label
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Extract predictions
        traits_pred = predictions['traits'].cpu().numpy()
        deception_prob = predictions['deception_prob'].item()
        
        # Extract targets
        traits_target = targets['personality'].cpu().numpy()
        deception_target = targets['deception'].item()
        
        # Compute trait errors
        trait_errors = np.abs(traits_pred - traits_target)
        trait_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        
        results = {
            'trait_errors': {name: float(err) for name, err in zip(trait_names, trait_errors)},
            'mean_trait_error': float(trait_errors.mean()),
            'deception_probability': float(deception_prob),
            'deception_label': int(deception_target),
            'deception_detected': int(deception_prob > 0.5)
        }
        
        if cultural_group:
            results['cultural_group'] = cultural_group
        
        return results
    
    def generate_hr_insights(
        self,
        predictions: Dict[str, torch.Tensor],
        confidence_threshold: float = 0.7
    ) -> Dict[str, any]:
        """
        Generate HR insights from predictions
        
        Args:
            predictions: Model predictions
            confidence_threshold: Confidence threshold for insights
            
        Returns:
            Dictionary of HR insights
        """
        traits = predictions['traits'].cpu().numpy()[0]  # Assuming batch size 1
        deception_prob = predictions['deception_prob'].item()
        confidence = predictions.get('confidence', None)
        
        trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        
        # Normalize traits to [0, 100] if needed
        if traits.max() <= 1.0:
            traits = traits * 100
        
        insights = {
            'personality_profile': {
                name: float(score) for name, score in zip(trait_names, traits)
            },
            'dominant_traits': [],
            'deception_risk': 'High' if deception_prob > 0.7 else 'Medium' if deception_prob > 0.4 else 'Low',
            'deception_probability': float(deception_prob),
            'recommendations': []
        }
        
        # Identify dominant traits (above 60)
        for i, (name, score) in enumerate(zip(trait_names, traits)):
            if score > 60:
                insights['dominant_traits'].append(name)
        
        # Generate recommendations
        if traits[1] > 70:  # High conscientiousness
            insights['recommendations'].append("Strong organizational skills and reliability")
        
        if traits[2] > 70:  # High extraversion
            insights['recommendations'].append("Good for team-oriented roles")
        
        if traits[4] > 60:  # High neuroticism
            insights['recommendations'].append("May benefit from stress management support")
        
        if deception_prob > 0.5:
            insights['recommendations'].append("Consider additional verification steps")
        
        return insights




















