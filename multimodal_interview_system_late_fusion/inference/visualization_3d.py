"""
3D Visualization Integration for HR Reports
Exports personality data and launches 3D visualization
"""
import json
import os
import subprocess
from typing import Dict, Optional
import numpy as np


class Personality3DVisualizer:
    """
    Integrates with PersonalityViz3D OpenGL application
    """
    
    def __init__(self, viz_exe_path: str = None, output_dir: str = "reports"):
        """
        Initialize 3D visualizer
        
        Args:
            viz_exe_path: Path to PersonalityViz3D.exe
            output_dir: Directory to save JSON files
        """
        # Try to find the executable
        if viz_exe_path is None:
            possible_paths = [
                r"D:\Code\Grad\PersonalityViz3D\x64\Debug\PersonalityViz3D.exe",
                r"D:\Code\PersonalityViz3D\x64\Debug\PersonalityViz3D.exe",
                r".\PersonalityViz3D.exe",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    viz_exe_path = path
                    break
        
        self.viz_exe_path = viz_exe_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_personality_json(
        self,
        traits: np.ndarray,
        deception_probability: float,
        candidate_name: Optional[str] = None,
        position: Optional[str] = None,
        confidence: Optional[np.ndarray] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export personality data to JSON for 3D visualization
        
        Args:
            traits: OCEAN trait scores (5 values, 0-100 scale)
            deception_probability: Deception probability (0-1)
            candidate_name: Candidate name
            position: Position applied for
            confidence: Confidence scores for each trait
            output_path: Custom output path
            
        Returns:
            Path to saved JSON file
        """
        # Normalize traits to 0-100 scale if needed
        if isinstance(traits, np.ndarray):
            traits = traits.tolist()
        
        if max(traits) <= 1.0:
            traits = [t * 100 for t in traits]
        
        trait_names = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        
        data = {
            "candidate_name": candidate_name or "Unknown",
            "position": position or "N/A",
            "traits": {name: float(value) for name, value in zip(trait_names, traits)},
            "deception_probability": float(deception_probability)
        }
        
        # Add confidence if available
        if confidence is not None:
            if isinstance(confidence, np.ndarray):
                confidence = confidence.tolist()
            data["confidence"] = {name: float(value) for name, value in zip(trait_names, confidence)}
        
        # Determine output path
        if output_path is None:
            candidate_id = (candidate_name or "candidate").replace(" ", "_").lower()
            output_path = os.path.join(self.output_dir, f"{candidate_id}_personality.json")
        
        # Save JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        print(f"Personality data exported to: {output_path}")
        return output_path
    
    def launch_visualization(
        self,
        json_path: Optional[str] = None,
        traits: Optional[np.ndarray] = None,
        candidate_name: Optional[str] = None,
        wait: bool = False
    ) -> bool:
        """
        Launch 3D visualization
        
        Args:
            json_path: Path to JSON file with personality data
            traits: Direct trait values (alternative to JSON)
            candidate_name: Candidate name
            wait: Whether to wait for visualization to close
            
        Returns:
            True if launched successfully
        """
        if self.viz_exe_path is None or not os.path.exists(self.viz_exe_path):
            print("ERROR: PersonalityViz3D.exe not found!")
            print("Please build the project first or specify the correct path.")
            return False
        
        # Build command
        cmd = [self.viz_exe_path]
        
        if json_path and os.path.exists(json_path):
            cmd.extend(["--json", json_path])
        elif traits is not None:
            if isinstance(traits, np.ndarray):
                traits = traits.tolist()
            if max(traits) <= 1.0:
                traits = [t * 100 for t in traits]
            
            cmd.extend(["-o", str(traits[0])])
            cmd.extend(["-c", str(traits[1])])
            cmd.extend(["-e", str(traits[2])])
            cmd.extend(["-a", str(traits[3])])
            cmd.extend(["-n", str(traits[4])])
        
        if candidate_name:
            cmd.extend(["--name", candidate_name])
        
        print(f"Launching 3D visualization: {' '.join(cmd)}")
        
        try:
            if wait:
                subprocess.run(cmd, cwd=os.path.dirname(self.viz_exe_path))
            else:
                subprocess.Popen(cmd, cwd=os.path.dirname(self.viz_exe_path))
            return True
        except Exception as e:
            print(f"Error launching visualization: {e}")
            return False
    
    def visualize_analysis(
        self,
        analysis_results: Dict,
        candidate_name: Optional[str] = None,
        position: Optional[str] = None,
        auto_launch: bool = True
    ) -> str:
        """
        Complete workflow: export data and optionally launch visualization
        
        Args:
            analysis_results: Results from inference pipeline
            candidate_name: Candidate name
            position: Position applied for
            auto_launch: Whether to automatically launch visualization
            
        Returns:
            Path to exported JSON file
        """
        traits = analysis_results['traits']
        deception_prob = analysis_results['deception_probability']
        confidence = analysis_results.get('confidence')
        
        # Export JSON
        json_path = self.export_personality_json(
            traits=traits,
            deception_probability=deception_prob,
            candidate_name=candidate_name,
            position=position,
            confidence=confidence
        )
        
        # Copy to PersonalityViz3D directory for auto-loading
        viz_dir = os.path.dirname(self.viz_exe_path) if self.viz_exe_path else None
        if viz_dir and os.path.exists(viz_dir):
            default_json_path = os.path.join(viz_dir, "personality_data.json")
            with open(json_path, 'r') as src:
                with open(default_json_path, 'w') as dst:
                    dst.write(src.read())
            print(f"Copied to: {default_json_path}")
        
        # Launch visualization
        if auto_launch:
            self.launch_visualization(json_path=json_path, candidate_name=candidate_name)
        
        return json_path


def integrate_with_report(report_generator, visualizer_3d):
    """
    Integrate 3D visualization with existing report generator
    
    Args:
        report_generator: HRReportGenerator instance
        visualizer_3d: Personality3DVisualizer instance
    """
    original_generate = report_generator.generate_report
    
    def enhanced_generate_report(
        analysis_results,
        candidate_name=None,
        position=None,
        save_html=True,
        save_pdf=False,
        show_3d=True  # New parameter
    ):
        # Generate original report
        report_path = original_generate(
            analysis_results=analysis_results,
            candidate_name=candidate_name,
            position=position,
            save_html=save_html,
            save_pdf=save_pdf
        )
        
        # Generate 3D visualization
        if show_3d:
            visualizer_3d.visualize_analysis(
                analysis_results=analysis_results,
                candidate_name=candidate_name,
                position=position,
                auto_launch=True
            )
        
        return report_path
    
    report_generator.generate_report = enhanced_generate_report
    return report_generator


# Example usage
if __name__ == "__main__":
    # Test with sample data
    visualizer = Personality3DVisualizer()
    
    # Sample analysis results
    sample_results = {
        'traits': np.array([75.0, 60.0, 80.0, 55.0, 40.0]),
        'deception_probability': 0.25,
        'confidence': np.array([0.85, 0.82, 0.90, 0.78, 0.88])
    }
    
    # Export and visualize
    json_path = visualizer.visualize_analysis(
        analysis_results=sample_results,
        candidate_name="John Doe",
        position="Software Engineer",
        auto_launch=True
    )
    
    print(f"\nExported to: {json_path}")
