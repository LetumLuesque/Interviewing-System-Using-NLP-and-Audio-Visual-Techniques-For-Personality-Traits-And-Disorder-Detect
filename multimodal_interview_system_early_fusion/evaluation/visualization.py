"""
Visualization module for interview analysis results
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import torch


class InterviewVisualizer:
    """
    Visualization tools for interview analysis
    """
    
    def __init__(self):
        pass
    
    def plot_big_five_profile(
        self,
        traits: np.ndarray,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot Big Five personality profile
        
        Args:
            traits: Trait scores [5] (OCEAN)
            save_path: Optional path to save figure
            
        Returns:
            Plotly figure
        """
        trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        
        # Normalize to [0, 100] if needed
        if traits.max() <= 1.0:
            traits = traits * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=trait_names,
            y=traits,
            marker_color='steelblue',
            text=[f'{t:.1f}' for t in traits],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Big Five Personality Profile',
            xaxis_title='Personality Traits',
            yaxis_title='Score (0-100)',
            yaxis_range=[0, 100],
            template='plotly_white',
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_3d_trait_space(
        self,
        traits: np.ndarray,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot 3D trait space visualization
        
        Args:
            traits: Trait scores [5] (OCEAN)
            save_path: Optional path to save figure
            
        Returns:
            Plotly 3D figure
        """
        trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        
        # Normalize to [0, 100] if needed
        if traits.max() <= 1.0:
            traits = traits * 100
        
        # Use first 3 traits for 3D plot (can be customized)
        fig = go.Figure(data=go.Scatter3d(
            x=[traits[0]],  # Openness
            y=[traits[1]],  # Conscientiousness
            z=[traits[2]],  # Extraversion
            mode='markers',
            marker=dict(
                size=20,
                color=traits[3],  # Agreeableness as color
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Agreeableness')
            ),
            text=[f'O: {traits[0]:.1f}, C: {traits[1]:.1f}, E: {traits[2]:.1f}'],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        fig.update_layout(
            title='3D Personality Trait Space',
            scene=dict(
                xaxis_title='Openness',
                yaxis_title='Conscientiousness',
                zaxis_title='Extraversion',
                xaxis_range=[0, 100],
                yaxis_range=[0, 100],
                zaxis_range=[0, 100]
            ),
            template='plotly_white',
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_deception_analysis(
        self,
        deception_prob: float,
        confidence: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot deception probability indicator
        
        Args:
            deception_prob: Deception probability [0, 1]
            confidence: Optional confidence score
            save_path: Optional path to save figure
            
        Returns:
            Plotly figure
        """
        # Determine risk level
        if deception_prob < 0.3:
            risk_level = 'Low'
            color = 'green'
        elif deception_prob < 0.7:
            risk_level = 'Medium'
            color = 'orange'
        else:
            risk_level = 'High'
            color = 'red'
        
        fig = go.Figure()
        
        # Gauge chart
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=deception_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Deception Risk: {risk_level}"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            title='Deception Detection Analysis',
            template='plotly_white',
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_trait_comparison(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot comparison between predictions and targets
        
        Args:
            predictions: Predicted traits [5]
            targets: Target traits [5]
            save_path: Optional path to save figure
            
        Returns:
            Plotly figure
        """
        trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        
        # Normalize if needed
        if predictions.max() <= 1.0:
            predictions = predictions * 100
        if targets.max() <= 1.0:
            targets = targets * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=trait_names,
            y=predictions,
            name='Predicted',
            marker_color='steelblue'
        ))
        
        fig.add_trace(go.Bar(
            x=trait_names,
            y=targets,
            name='Target',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Predicted vs Target Traits',
            xaxis_title='Personality Traits',
            yaxis_title='Score (0-100)',
            barmode='group',
            template='plotly_white',
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_comprehensive_report(
        self,
        traits: np.ndarray,
        deception_prob: float,
        confidence: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive visualization report
        
        Args:
            traits: Trait scores [5]
            deception_prob: Deception probability
            confidence: Optional confidence scores [5]
            save_path: Optional path to save figure
            
        Returns:
            Plotly figure with subplots
        """
        # Normalize if needed
        if isinstance(traits, torch.Tensor):
            traits = traits.cpu().numpy()
        if traits.max() <= 1.0:
            traits = traits * 100
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Big Five Profile', '3D Trait Space', 'Deception Risk', 'Confidence Scores'),
            specs=[[{"type": "bar"}, {"type": "scatter3d"}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        
        # Big Five Profile
        fig.add_trace(
            go.Bar(x=trait_names, y=traits, name='Traits', marker_color='steelblue'),
            row=1, col=1
        )
        
        # 3D Trait Space
        fig.add_trace(
            go.Scatter3d(
                x=[traits[0]],
                y=[traits[1]],
                z=[traits[2]],
                mode='markers',
                marker=dict(size=20, color=traits[3], colorscale='Viridis'),
                name='Trait Space'
            ),
            row=1, col=2
        )
        
        # Deception Risk
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=deception_prob * 100,
                title={'text': "Deception Risk"},
                gauge={'axis': {'range': [None, 100]}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=1
        )
        
        # Confidence Scores
        if confidence is not None:
            if isinstance(confidence, torch.Tensor):
                confidence = confidence.cpu().numpy()
            fig.add_trace(
                go.Bar(x=trait_names, y=confidence * 100, name='Confidence', marker_color='green'),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Comprehensive Interview Analysis Report',
            height=1000,
            template='plotly_white',
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
































