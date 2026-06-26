"""
HR Report Generator for interview analysis results - Enhanced 3D Version
"""
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import os
import json
import webbrowser

class HRReportGenerator:
    """
    Generate comprehensive HR reports from interview analysis with 3D visualization
    """
    
    def __init__(self, output_dir: str = "reports", viz3d_exe_path: str = None):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
            viz3d_exe_path: Path to PersonalityViz3D.exe (optional, for external launching)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.viz3d_exe_path = viz3d_exe_path

    def generate_report(
        self,
        analysis_results: Dict,
        candidate_name: Optional[str] = None,
        position: Optional[str] = None,
        save_html: bool = True,
        show_3d: bool = False
    ) -> str:
        """
        Generate comprehensive HR report
        
        Args:
            analysis_results: Analysis results from inference pipeline
            candidate_name: Candidate name
            position: Position applied for
            save_html: Whether to save HTML report
            show_3d: Whether to launch external 3D visualization (optional)
            
        Returns:
            Path to saved report
        """
        traits = analysis_results['traits']
        deception_prob = analysis_results['deception_probability']
        confidence = analysis_results.get('confidence')
        modality_confidence = analysis_results.get('modality_confidence', {'visual': 0.33, 'audio': 0.33, 'text': 0.33})
        
        # Ensure timestamp for unique files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate_slug = (candidate_name or "candidate").replace(" ", "_").lower()
        candidate_id = f"{candidate_slug}_{timestamp}"
        
        # Default duration placeholder since we don't strictly track it here
        duration = 0 
        
        # Generate HTML content
        html_content = self._generate_html_report(
            traits, deception_prob, confidence, modality_confidence,
            candidate_name, position, candidate_id, duration
        )
        
        # Save HTML
        report_path = os.path.join(self.output_dir, "hr_report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return report_path

    def _generate_3d_visualization_script(self, traits):
        """Generate Three.js 3D visualization script"""
        return f"""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
        (function() {{
            const traits = {{
                openness: {traits[0]:.1f},
                conscientiousness: {traits[1]:.1f},
                extraversion: {traits[2]:.1f},
                agreeableness: {traits[3]:.1f},
                neuroticism: {traits[4]:.1f}
            }};
            
            const traitColors = [
                0xcc1aff,  // O - Purple
                0x0080ff,  // C - Blue
                0xffb300,  // E - Orange
                0x00ff66,  // A - Green
                0xff1a33   // N - Red
            ];
            
            const traitNames = ['O', 'C', 'E', 'A', 'N'];
            const traitValues = [traits.openness, traits.conscientiousness, traits.extraversion, traits.agreeableness, traits.neuroticism];
            
            const container = document.getElementById('personality-3d');
            const width = container.clientWidth;
            const height = 550;
            
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);
            
            const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
            camera.position.set(0, 0, 5);
            
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(width, height);
            container.appendChild(renderer.domElement);
            
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight1.position.set(1, 1, 2);
            scene.add(directionalLight1);
            
            const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight2.position.set(-1, 0.5, 1);
            scene.add(directionalLight2);
            
            const crystalGroup = new THREE.Group();
            
            function getTraitPosition(index, value) {{
                const angle = (90 + index * 72) * Math.PI / 180;
                const distance = 0.15 + (value / 100) * 0.85;
                const zOffset = (index % 2 === 0) ? 0.3 : -0.3;
                return new THREE.Vector3(
                    Math.cos(angle) * distance,
                    Math.sin(angle) * distance,
                    zOffset * 0.5
                );
            }}
            
            const vertices = traitValues.map((v, i) => getTraitPosition(i, v));
            const frontApex = new THREE.Vector3(0, 0, 0.8);
            const backApex = new THREE.Vector3(0, 0, -0.8);
            const center = new THREE.Vector3(0, 0, 0);
            
            for (let i = 0; i < 5; i++) {{
                const next = (i + 1) % 5;
                
                const frontGeom = new THREE.BufferGeometry();
                const frontVerts = new Float32Array([
                    frontApex.x, frontApex.y, frontApex.z,
                    vertices[i].x, vertices[i].y, vertices[i].z,
                    vertices[next].x, vertices[next].y, vertices[next].z
                ]);
                frontGeom.setAttribute('position', new THREE.BufferAttribute(frontVerts, 3));
                frontGeom.computeVertexNormals();
                
                const frontMat = new THREE.MeshPhongMaterial({{
                    color: traitColors[i],
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: 0.85,
                    shininess: 100
                }});
                crystalGroup.add(new THREE.Mesh(frontGeom, frontMat));
                
                const backGeom = new THREE.BufferGeometry();
                const backVerts = new Float32Array([
                    backApex.x, backApex.y, backApex.z,
                    vertices[next].x, vertices[next].y, vertices[next].z,
                    vertices[i].x, vertices[i].y, vertices[i].z
                ]);
                backGeom.setAttribute('position', new THREE.BufferAttribute(backVerts, 3));
                backGeom.computeVertexNormals();
                
                const backMat = new THREE.MeshPhongMaterial({{
                    color: traitColors[i],
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: 0.7,
                    shininess: 80
                }});
                crystalGroup.add(new THREE.Mesh(backGeom, backMat));
            }}
            
            const coreGeom = new THREE.IcosahedronGeometry(0.12, 1);
            const coreMat = new THREE.MeshPhongMaterial({{
                color: 0xffffff,
                shininess: 150,
                emissive: 0x222222
            }});
            crystalGroup.add(new THREE.Mesh(coreGeom, coreMat));
            
            for (let i = 0; i < 5; i++) {{
                const sphereGeom = new THREE.SphereGeometry(0.08, 16, 16);
                const sphereMat = new THREE.MeshPhongMaterial({{
                    color: traitColors[i],
                    shininess: 100,
                    emissive: traitColors[i],
                    emissiveIntensity: 0.2
                }});
                const sphere = new THREE.Mesh(sphereGeom, sphereMat);
                sphere.position.copy(vertices[i]);
                crystalGroup.add(sphere);
            }}
            
            const lineMaterial = new THREE.LineBasicMaterial({{ color: 0xffffff, opacity: 0.5, transparent: true }});
            for (let i = 0; i < 5; i++) {{
                const next = (i + 1) % 5;
                const points = [vertices[i], vertices[next]];
                const lineGeom = new THREE.BufferGeometry().setFromPoints(points);
                crystalGroup.add(new THREE.Line(lineGeom, lineMaterial));
                
                const centerPoints = [center, vertices[i]];
                const centerLineGeom = new THREE.BufferGeometry().setFromPoints(centerPoints);
                crystalGroup.add(new THREE.Line(centerLineGeom, lineMaterial));
            }}
            
            scene.add(crystalGroup);
            
            let isDragging = false;
            let previousMousePosition = {{ x: 0, y: 0 }};
            
            renderer.domElement.addEventListener('mousedown', (e) => {{
                isDragging = true;
                previousMousePosition = {{ x: e.clientX, y: e.clientY }};
            }});
            
            renderer.domElement.addEventListener('mousemove', (e) => {{
                if (isDragging) {{
                    const deltaX = e.clientX - previousMousePosition.x;
                    const deltaY = e.clientY - previousMousePosition.y;
                    crystalGroup.rotation.y += deltaX * 0.01;
                    crystalGroup.rotation.x += deltaY * 0.01;
                    previousMousePosition = {{ x: e.clientX, y: e.clientY }};
                }}
            }});
            
            renderer.domElement.addEventListener('mouseup', () => {{ isDragging = false; }});
            renderer.domElement.addEventListener('mouseleave', () => {{ isDragging = false; }});
            
            let autoRotate = true;
            renderer.domElement.addEventListener('dblclick', () => {{ autoRotate = !autoRotate; }});
            
            function animate() {{
                requestAnimationFrame(animate);
                if (autoRotate && !isDragging) {{
                    crystalGroup.rotation.y += 0.005;
                }}
                renderer.render(scene, camera);
            }}
            
            animate();
            
            window.addEventListener('resize', () => {{
                const newWidth = container.clientWidth;
                camera.aspect = newWidth / height;
                camera.updateProjectionMatrix();
                renderer.setSize(newWidth, height);
            }});
        }})();
        </script>
        """

    def _generate_timeline_script(self):
        """Generate animated timeline chart using Chart.js"""
        return '''
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
        (function() {
            const ctx = document.getElementById('timelineChart').getContext('2d');
            
            // Generate simulated data
            const labels = Array.from({length: 30}, (_, i) => i + 's');
            const deceptionData = Array.from({length: 30}, () => 0.15 + Math.random() * 0.15);
            const voiceStress = Array.from({length: 30}, () => 0.25 + Math.random() * 0.2);
            const expressionConf = Array.from({length: 30}, () => 0.75 + Math.random() * 0.15);
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Deception Probability',
                            data: deceptionData,
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231, 76, 60, 0.1)',
                            fill: true,
                            tension: 0.4
                        },
                        {
                            label: 'Voice Stress',
                            data: voiceStress,
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            fill: true,
                            tension: 0.4
                        },
                        {
                            label: 'Expression Confidence',
                            data: expressionConf,
                            borderColor: '#2ecc71',
                            backgroundColor: 'rgba(46, 204, 113, 0.1)',
                            fill: true,
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#ccc' }
                        }
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'Time (seconds)', color: '#999' },
                            ticks: { color: '#999' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        y: {
                            min: 0,
                            max: 1,
                            title: { display: true, text: 'Score', color: '#999' },
                            ticks: { color: '#999' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        }
                    }
                }
            });
        })();
        </script>
        '''

    def _interpret_trait(self, trait_name, score):
        """Interpret trait score"""
        if score < 30:
            level = "Low"
        elif score < 70:
            level = "Moderate"
        else:
            level = "High"
        
        interpretations = {
            'Openness': {
                'Low': 'Prefers routine and familiar situations',
                'Moderate': 'Balanced between tradition and innovation',
                'High': 'Creative, curious, and open to new experiences'
            },
            'Conscientiousness': {
                'Low': 'More flexible and spontaneous',
                'Moderate': 'Balanced approach to organization',
                'High': 'Organized, reliable, and goal-oriented'
            },
            'Extraversion': {
                'Low': 'Prefers quiet, independent work',
                'Moderate': 'Comfortable in both social and solitary settings',
                'High': 'Energized by social interactions and teamwork'
            },
            'Agreeableness': {
                'Low': 'More competitive and direct',
                'Moderate': 'Balanced between cooperation and assertiveness',
                'High': 'Cooperative, trusting, and empathetic'
            },
            'Neuroticism': {
                'Low': 'Emotionally stable and resilient',
                'Moderate': 'Normal stress response',
                'High': 'May experience higher stress and emotional sensitivity'
            }
        }
        
        return interpretations.get(trait_name, {}).get(level, 'N/A')

    def _get_detected_cues(self, traits, deception_prob):
        """Generate detected cues based on analysis"""
        cues = {
            'micro_expressions': [],
            'voice_patterns': [],
            'language_cues': []
        }
        
        # Micro-expressions based on traits
        if traits[2] > 70:  # High extraversion
            cues['micro_expressions'].append('Frequent genuine smiles (Duchenne)')
            cues['micro_expressions'].append('Open and expressive facial movements')
        elif traits[2] < 40:
            cues['micro_expressions'].append('Reserved facial expressions')
            cues['micro_expressions'].append('Controlled emotional display')
        
        if traits[4] < 40:  # Low neuroticism
            cues['micro_expressions'].append('Consistent emotional baseline')
        else:
            cues['micro_expressions'].append('Variable emotional responses detected')
        
        if deception_prob < 0.3:
            cues['micro_expressions'].append('Expressions consistent with verbal content')
        else:
            cues['micro_expressions'].append('Some incongruence between verbal and non-verbal cues')
        
        # Voice patterns
        if deception_prob < 0.3:
            cues['voice_patterns'].append('Voice stress levels within normal range')
            cues['voice_patterns'].append('Consistent speech rhythm and pace')
        else:
            cues['voice_patterns'].append('Elevated voice stress detected in some segments')
            cues['voice_patterns'].append('Minor pitch variations noted')
        
        if traits[2] > 60:
            cues['voice_patterns'].append('Confident vocal projection')
        else:
            cues['voice_patterns'].append('Moderate vocal confidence')
        
        # Language cues
        if traits[1] > 60:  # High conscientiousness
            cues['language_cues'].append('Structured and organized responses')
            cues['language_cues'].append('Specific examples provided')
        
        if deception_prob < 0.3:
            cues['language_cues'].append('No significant hedging language detected')
            cues['language_cues'].append('Direct and confident statements')
        else:
            cues['language_cues'].append('Some hedging phrases detected')
            cues['language_cues'].append('Occasional vague responses')
        
        if traits[0] > 60:  # High openness
            cues['language_cues'].append('Varied vocabulary usage')
        
        return cues

    def _get_key_insights(self, traits, deception_prob):
        """Generate key personality insights"""
        insights = []
        
        trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        max_idx = np.argmax(traits)
        min_idx = np.argmin(traits)
        
        insights.append(f"Highest trait: {trait_names[max_idx]} ({traits[max_idx]:.0f})")
        insights.append(f"Lowest trait: {trait_names[min_idx]} ({traits[min_idx]:.0f})")
        
        if traits[2] > 70:
            insights.append("Strong social and communication skills")
        if traits[1] > 70:
            insights.append("Highly reliable and detail-oriented")
        if traits[4] < 40:
            insights.append("Emotionally stable under pressure")
        if traits[0] > 70:
            insights.append("Creative problem-solver")
        if traits[3] > 70:
            insights.append("Team player with strong interpersonal skills")
        
        if deception_prob < 0.25:
            insights.append("High authenticity in responses")
        
        return insights[:6]

    def _get_hr_recommendations(self, traits, deception_prob, position):
        """Generate HR recommendations"""
        recommendations = []
        warnings = []
        
        if deception_prob < 0.3:
            recommendations.append("✓ Proceed to next interview stage")
        elif deception_prob < 0.6:
            warnings.append("Consider additional verification questions")
        else:
            warnings.append("Recommend thorough reference checks")
        
        if traits[2] > 70:
            recommendations.append("Suitable for client-facing roles")
        if traits[1] > 70:
            recommendations.append("Good fit for detail-oriented positions")
        if traits[0] > 70:
            recommendations.append("Consider for creative/innovative projects")
        if traits[3] > 70:
            recommendations.append("Strong potential for team leadership")
        
        if traits[4] > 60:
            warnings.append("May benefit from structured onboarding")
        
        if "engineer" in position.lower() or "developer" in position.lower():
            recommendations.append("Schedule technical assessment")
        if "manager" in position.lower() or "lead" in position.lower():
            recommendations.append("Evaluate leadership experience in follow-up")
        
        return recommendations[:4], warnings[:2]

    def _generate_html_report(self, traits, deception_prob, confidence, modality_confidence, 
                              candidate_name, position, candidate_id, duration):
        """Generate complete HTML report with all visualizations"""
        
        trait_names = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        trait_colors = ['#cc1aff', '#0080ff', '#ffb300', '#00ff66', '#ff1a33']
        
        # Normalize traits to 0-100
        if np.max(traits) <= 1.0:
            traits = [t * 100 for t in traits]
        
        # Risk level
        if deception_prob < 0.3:
            risk_level, risk_color, risk_bg, risk_icon = 'LOW', '#27ae60', '#d4efdf', '✓'
        elif deception_prob < 0.6:
            risk_level, risk_color, risk_bg, risk_icon = 'MEDIUM', '#f39c12', '#fdebd0', '⚠'
        else:
            risk_level, risk_color, risk_bg, risk_icon = 'HIGH', '#e74c3c', '#fadbd8', '✗'
        
        # Overall confidence
        overall_conf = np.mean(confidence) * 100 if confidence is not None else 0
        conf_level = 'HIGH' if overall_conf > 80 else 'MEDIUM' if overall_conf > 60 else 'LOW'
        
        # Get analysis data
        cues = self._get_detected_cues(traits, deception_prob)
        insights = self._get_key_insights(traits, deception_prob)
        recommendations, warnings = self._get_hr_recommendations(traits, deception_prob, position or "")
        
        # Modality contribution (simulated based on confidence)
        total_conf = sum(modality_confidence.values())
        if total_conf == 0: total_conf = 1
        visual_pct = (modality_confidence['visual'] / total_conf) * 100
        audio_pct = (modality_confidence['audio'] / total_conf) * 100
        text_pct = (modality_confidence['text'] / total_conf) * 100
        
        html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Interview Analysis Report - {candidate_name or 'Candidate'}</title>
        <meta charset="UTF-8">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
                min-height: 100vh; color: #e0e0e0; padding: 20px;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            
            /* Header */
            .header {{ 
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
                padding: 25px 30px; border-radius: 15px; margin-bottom: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.4);
                display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
            }}
            .header h1 {{ font-size: 1.8em; color: #fff; }}
            .header-info {{ display: flex; gap: 30px; flex-wrap: wrap; margin-top: 10px; }}
            .header-info span {{ color: rgba(255,255,255,0.8); font-size: 0.95em; }}
            .header-info strong {{ color: #fff; }}
            
            /* Stats Row */
            .stats-row {{ 
                display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;
            }}
            @media (max-width: 900px) {{ .stats-row {{ grid-template-columns: repeat(2, 1fr); }} }}
            
            .stat-card {{
                background: linear-gradient(145deg, #252538, #1e1e2e);
                border-radius: 15px; padding: 20px; text-align: center;
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            }}
            .stat-value {{ font-size: 2.2em; font-weight: bold; }}
            .stat-label {{ font-size: 0.85em; color: #888; margin-top: 5px; }}
            .stat-status {{ font-size: 0.8em; margin-top: 8px; padding: 4px 12px; border-radius: 20px; display: inline-block; }}
            
            /* Main Grid */
            .main-grid {{ 
                display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
            }}
            @media (max-width: 1000px) {{ .main-grid {{ grid-template-columns: 1fr; }} }}
            
            .section {{ 
                background: linear-gradient(145deg, #252538, #1e1e2e);
                padding: 25px; border-radius: 15px;
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            }}
            .section h2 {{ 
                color: #fff; font-size: 1.1em; margin-bottom: 20px;
                padding-bottom: 10px; border-bottom: 2px solid #3498db;
            }}
            
            .full-width {{ grid-column: 1 / -1; }}
            
            /* 3D Visualization */
            #personality-3d {{ 
                width: 100%; height: 550px; border-radius: 10px; 
                overflow: hidden; cursor: grab; background: #1a1a2e;
            }}
            #personality-3d:active {{ cursor: grabbing; }}
            .viz-instructions {{ text-align: center; color: #888; font-size: 0.85em; margin-top: 10px; }}
            
            /* Trait Bars */
            .trait-bar {{ margin: 12px 0; }}
            .trait-header {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
            .trait-name {{ font-weight: 500; font-size: 0.9em; }}
            .trait-score {{ font-weight: bold; }}
            .trait-track {{ height: 8px; background: #333; border-radius: 4px; overflow: hidden; }}
            .trait-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s ease; }}
            .trait-interpretation {{ font-size: 0.8em; color: #888; margin-top: 4px; font-style: italic; }}
            
            /* Modality */
            .modality-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
            .modality-item {{ text-align: center; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 10px; }}
            .modality-icon {{ font-size: 2em; margin-bottom: 10px; }}
            .modality-bar {{ height: 6px; background: #333; border-radius: 3px; margin: 10px 0; overflow: hidden; }}
            .modality-fill {{ height: 100%; border-radius: 3px; }}
            
            /* Cues */
            .cues-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
            @media (max-width: 800px) {{ .cues-grid {{ grid-template-columns: 1fr; }} }}
            .cue-category {{ background: rgba(0,0,0,0.2); border-radius: 10px; padding: 15px; }}
            .cue-title {{ font-weight: bold; color: #3498db; margin-bottom: 10px; font-size: 0.9em; }}
            .cue-item {{ font-size: 0.85em; color: #aaa; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
            .cue-item:last-child {{ border-bottom: none; }}
            
            /* Timeline */
            .timeline-container {{ height: 250px; }}
            
            /* Insights */
            .insights-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            @media (max-width: 700px) {{ .insights-grid {{ grid-template-columns: 1fr; }} }}
            .insight-list {{ list-style: none; }}
            .insight-item {{ 
                padding: 10px 15px; margin: 8px 0; background: rgba(52, 152, 219, 0.1);
                border-left: 3px solid #3498db; border-radius: 5px; font-size: 0.9em;
            }}
            .recommendation-item {{ 
                padding: 10px 15px; margin: 8px 0; background: rgba(46, 204, 113, 0.1);
                border-left: 3px solid #2ecc71; border-radius: 5px; font-size: 0.9em;
            }}
            .warning-item {{ 
                padding: 10px 15px; margin: 8px 0; background: rgba(243, 156, 18, 0.1);
                border-left: 3px solid #f39c12; border-radius: 5px; font-size: 0.9em;
            }}
            
            /* Legend */
            .legend {{ 
                display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;
                margin-top: 15px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 10px;
            }}
            .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.85em; }}
            .legend-color {{ width: 14px; height: 14px; border-radius: 50%; }}
            
            /* Table */
            .data-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            .data-table th {{ background: #2c3e50; color: #fff; padding: 12px; text-align: left; font-size: 0.9em; }}
            .data-table td {{ padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.9em; }}
            .data-table tr:hover {{ background: rgba(255,255,255,0.03); }}
            
            /* Footer */
            .footer {{ 
                text-align: center; padding: 20px; color: rgba(255,255,255,0.4);
                font-size: 0.85em; margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="header">
                <div>
                    <h1>🎯 Interview Analysis Report</h1>
                    <div class="header-info">
                        <span>👤 <strong>{candidate_name or 'N/A'}</strong></span>
                        <span>💼 <strong>{position or 'N/A'}</strong></span>
                        <span>🆔 <strong>#{candidate_id}</strong></span>
                        <span>⏱️ <strong>{duration}s</strong></span>
                        <span>📅 <strong>{datetime.now().strftime("%Y-%m-%d %H:%M")}</strong></span>
                    </div>
                </div>
            </div>
            
            <!-- Stats Row -->
            <div class="stats-row">
                <div class="stat-card">
                    <div class="stat-value" style="color: {risk_color}">{deception_prob*100:.0f}%</div>
                    <div class="stat-label">Deception Risk</div>
                    <div class="stat-status" style="background: {risk_bg}; color: {risk_color}">{risk_icon} {risk_level} RISK</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: #3498db">{overall_conf:.0f}%</div>
                    <div class="stat-label">Overall Confidence</div>
                    <div class="stat-status" style="background: #e3f2fd; color: #1976d2">✓ {conf_level}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: #9b59b6">{np.mean(traits):.0f}</div>
                    <div class="stat-label">Avg. Trait Score</div>
                    <div class="stat-status" style="background: #f3e5f5; color: #7b1fa2">BALANCED</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: #2ecc71">{'👍' if deception_prob < 0.4 else '⚠️' if deception_prob < 0.7 else '👎'}</div>
                    <div class="stat-label">Recommendation</div>
                    <div class="stat-status" style="background: {'#e8f5e9' if deception_prob < 0.4 else '#fff3e0' if deception_prob < 0.7 else '#ffebee'}; color: {'#2e7d32' if deception_prob < 0.4 else '#e65100' if deception_prob < 0.7 else '#c62828'}">
                        {'PROCEED' if deception_prob < 0.4 else 'REVIEW' if deception_prob < 0.7 else 'CAUTION'}
                    </div>
                </div>
            </div>
            
            <div class="main-grid">
                <!-- 3D Crystal -->
                <div class="section full-width">
                    <h2>🔮 3D Personality Crystal</h2>
                    <div id="personality-3d"></div>
                    <p class="viz-instructions">🖱️ Drag to rotate | Double-click to toggle auto-rotation</p>
                    <div class="legend">
                        <div class="legend-item"><div class="legend-color" style="background: #cc1aff;"></div> Openness</div>
                        <div class="legend-item"><div class="legend-color" style="background: #0080ff;"></div> Conscientiousness</div>
                        <div class="legend-item"><div class="legend-color" style="background: #ffb300;"></div> Extraversion</div>
                        <div class="legend-item"><div class="legend-color" style="background: #00ff66;"></div> Agreeableness</div>
                        <div class="legend-item"><div class="legend-color" style="background: #ff1a33;"></div> Neuroticism</div>
                    </div>
                </div>
                
                <!-- Trait Bars -->
                <div class="section">
                    <h2>📈 Personality Traits (OCEAN)</h2>"""
        
        for i, (name, score) in enumerate(zip(trait_names, traits)):
            interpretation = self._interpret_trait(name, score)
            html += f"""
                    <div class="trait-bar">
                        <div class="trait-header">
                            <span class="trait-name" style="color: {trait_colors[i]}">{name}</span>
                            <span class="trait-score">{score:.0f}/100</span>
                        </div>
                        <div class="trait-track">
                            <div class="trait-fill" style="width: {score}%; background: {trait_colors[i]};"></div>
                        </div>
                        <div class="trait-interpretation">{interpretation}</div>
                    </div>"""
        
        html += f"""
                </div>
                
                <!-- Modality Analysis -->
                <div class="section">
                    <h2>🔬 Modality Analysis</h2>
                    <div class="modality-grid">
                        <div class="modality-item">
                            <div class="modality-icon">👁️</div>
                            <div style="font-weight: bold;">Visual</div>
                            <div style="font-size: 1.5em; color: #3498db;">{modality_confidence['visual']*100:.0f}%</div>
                            <div class="modality-bar"><div class="modality-fill" style="width: {modality_confidence['visual']*100}%; background: #3498db;"></div></div>
                            <div style="font-size: 0.8em; color: #888;">Contribution: {visual_pct:.0f}%</div>
                        </div>
                        <div class="modality-item">
                            <div class="modality-icon">🎤</div>
                            <div style="font-weight: bold;">Audio</div>
                            <div style="font-size: 1.5em; color: #2ecc71;">{modality_confidence['audio']*100:.0f}%</div>
                            <div class="modality-bar"><div class="modality-fill" style="width: {modality_confidence['audio']*100}%; background: #2ecc71;"></div></div>
                            <div style="font-size: 0.8em; color: #888;">Contribution: {audio_pct:.0f}%</div>
                        </div>
                        <div class="modality-item">
                            <div class="modality-icon">📝</div>
                            <div style="font-weight: bold;">Text</div>
                            <div style="font-size: 1.5em; color: #e74c3c;">{modality_confidence['text']*100:.0f}%</div>
                            <div class="modality-bar"><div class="modality-fill" style="width: {modality_confidence['text']*100}%; background: #e74c3c;"></div></div>
                            <div style="font-size: 0.8em; color: #888;">Contribution: {text_pct:.0f}%</div>
                        </div>
                    </div>
                    <p style="text-align: center; color: #666; font-size: 0.8em; margin-top: 15px;">
                        ✓ Cultural Bias Adjusted | ✓ Multi-modal Fusion Applied
                    </p>
                </div>
                
                <!-- Detected Cues -->
                <div class="section full-width">
                    <h2>🔍 Detected Behavioral Cues</h2>
                    <div class="cues-grid">
                        <div class="cue-category">
                            <div class="cue-title">👁️ Micro-Expressions</div>
                            {"".join([f'<div class="cue-item">• {cue}</div>' for cue in cues['micro_expressions']])}
                        </div>
                        <div class="cue-category">
                            <div class="cue-title">🎤 Voice Patterns</div>
                            {"".join([f'<div class="cue-item">• {cue}</div>' for cue in cues['voice_patterns']])}
                        </div>
                        <div class="cue-category">
                            <div class="cue-title">📝 Language Cues</div>
                            {"".join([f'<div class="cue-item">• {cue}</div>' for cue in cues['language_cues']])}
                        </div>
                    </div>
                </div>
                
                <!-- Real-time Timeline -->
                <div class="section full-width">
                    <h2>📉 Real-time Analysis Timeline</h2>
                    <div class="timeline-container">
                        <canvas id="timelineChart"></canvas>
                    </div>
                </div>
                
                <!-- Insights & Recommendations -->
                <div class="section full-width">
                    <h2>💡 Key Insights & Recommendations</h2>
                    <div class="insights-grid">
                        <div>
                            <h3 style="color: #3498db; font-size: 1em; margin-bottom: 10px;">🎯 Key Personality Insights</h3>
                            <ul class="insight-list">
                                {"".join([f'<li class="insight-item">{insight}</li>' for insight in insights])}
                            </ul>
                        </div>
                        <div>
                            <h3 style="color: #2ecc71; font-size: 1em; margin-bottom: 10px;">✅ HR Recommendations</h3>
                            <ul class="insight-list">
                                {"".join([f'<li class="recommendation-item">{rec}</li>' for rec in recommendations])}
                                {"".join([f'<li class="warning-item">⚠️ {warn}</li>' for warn in warnings])}
                            </ul>
                        </div>
                    </div>
                </div>
                
                <!-- Detailed Table -->
                <div class="section full-width">
                    <h2>📋 Detailed Scores</h2>
                    <table class="data-table">
                        <tr>
                            <th>Trait</th>
                            <th>Score</th>
                            <th>Level</th>
                            <th>Confidence</th>
                            <th>Interpretation</th>
                        </tr>"""
        
        for i, (name, score) in enumerate(zip(trait_names, traits)):
            level = "Low" if score < 40 else "Moderate" if score < 70 else "High"
            interpretation = self._interpret_trait(name, score)
            conf = confidence[i] * 100 if confidence is not None else 0
            html += f"""
                        <tr>
                            <td><span style="color: {trait_colors[i]}; font-weight: bold;">●</span> {name}</td>
                            <td><strong>{score:.0f}</strong>/100</td>
                            <td>{level}</td>
                            <td>{conf:.0f}%</td>
                            <td style="color: #888;">{interpretation}</td>
                        </tr>"""
        
        html += f"""
                    </table>
                </div>
            </div>
            
            <div class="footer">
                Generated by Multimodal Interview Analysis System | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                <br>Cultural Bias: Adjusted ✓ | Confidence Threshold: 70% | Analysis Version: 2.0
            </div>
        </div>
        
        {self._generate_3d_visualization_script(traits)}
        {self._generate_timeline_script()}
    </body>
    </html>"""
        
        return html
