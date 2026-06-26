import json
import os
import random
from pathlib import Path
from datetime import datetime

# ── Course Loader ─────────────────────────────────────────────────────────────
BASE_DIR = Path(r"D:\Code\Grad\authentic_growth_reports")
COURSES_FILE = BASE_DIR / "edx_courses.json"

with open(COURSES_FILE, 'r', encoding='utf-8') as f:
    ALL_COURSES = json.load(f)

# Group courses by category for quick lookup
COURSES_BY_CAT = {}
for c in ALL_COURSES:
    COURSES_BY_CAT.setdefault(c["category"], []).append(c)

class DatasetGenerator:
    def __init__(self, template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()
            
    def _interpret_trait(self, score, low_desc, mod_desc, high_desc):
        if score < 30:
            return "Low", low_desc
        elif score < 70:
            return "Moderate", mod_desc
        else:
            return "High", high_desc

    def select_courses(self, traits, dec_prob):
        # Determine coaching needs
        needs = []
        
        # traits are in order: [Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism]
        o_score, c_score, e_score, a_score, n_score = traits
        
        if o_score < 50:
            needs.append(("creativity", 50 - o_score))
        if c_score < 50:
            needs.append(("organization", 50 - c_score))
        if e_score < 50:
            needs.append(("communication", 50 - e_score))
        if a_score < 50:
            needs.append(("teamwork", 50 - a_score))
        if n_score > 50:
            needs.append(("resilience", n_score - 50))
        if dec_prob > 0.3:
            needs.append(("honesty", dec_prob * 100))
            
        # Add general leadership and strategy as defaults if not enough needs
        needs.append(("leadership", 10))
        needs.append(("strategy", 5))
        
        # Sort by urgency (highest delta)
        needs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 2 unique categories
        selected_courses = []
        selected_cats = set()
        
        for cat, delta in needs:
            if cat in COURSES_BY_CAT and cat not in selected_cats:
                # Select a course from this category
                possible = COURSES_BY_CAT[cat]
                # To keep it deterministic for generation but varied, we can hash the traits
                idx = int(sum(traits)) % len(possible)
                selected_courses.append(possible[idx])
                selected_cats.add(cat)
            if len(selected_courses) == 2:
                break
                
        # Fallback if we somehow don't have 2
        while len(selected_courses) < 2:
            for cat in COURSES_BY_CAT:
                if cat not in selected_cats:
                    selected_courses.append(COURSES_BY_CAT[cat][0])
                    selected_cats.add(cat)
                    break
                    
        return selected_courses

    def process_metrics(self, data):
        traits = data.get('traits', [0]*5)
        # Normalize if needed (0-1 to 0-100)
        if max(traits) <= 1.1:
            traits = [t * 100 for t in traits]
            
        dec_prob = data.get('deception_probability', 0)
        risk_level = "HIGH" if dec_prob > 0.6 else "MEDIUM" if dec_prob > 0.3 else "LOW"
        
        meta = data.get('metadata', {})
        candidate_name = meta.get('candidate_name', "N/A")
        position = meta.get('position', "N/A")
        
        # Working Style Interpretations (non-technical)
        o_lvl, o_interp = self._interpret_trait(
            traits[0], 
            "Prefers routine and familiar situations over new approaches", 
            "Balanced between traditional methods and open-minded innovation", 
            "Highly creative, imaginative, and receptive to new ideas"
        )
        c_lvl, c_interp = self._interpret_trait(
            traits[1], 
            "Prefers a flexible, spontaneous approach to tasks rather than rigid planning", 
            "Demonstrates a balanced approach to task organization and schedules", 
            "Highly organized, methodical, and strongly goal-oriented"
        )
        e_lvl, e_interp = self._interpret_trait(
            traits[2], 
            "Tends to work best in quiet, independent environments", 
            "Comfortable in both social settings and quiet individual workspaces", 
            "Energized by collaborative activities and social interactions"
        )
        a_lvl, a_interp = self._interpret_trait(
            traits[3], 
            "Values directness and competitive drive in communication", 
            "Balances personal assertiveness with team collaboration", 
            "Highly cooperative, supportive, and empathetic toward team members"
        )
        # Neuroticism -> Resilience (inverted logic for interpretations)
        n_lvl, n_interp = self._interpret_trait(
            traits[4], 
            "Highly stable, calm, and resilient under high pressure", 
            "Maintains a typical, balanced response to workplace stress", 
            "Sensitive to high-pressure environments; thrives with extra support"
        )
        
        # Select 2 courses based on scores
        rec_courses = self.select_courses(traits, dec_prob)
        
        # Build mapping
        mapping = {
            "candidate_name": candidate_name,
            "candidate_id": meta.get('candidate_id', "SYNTH_" + str(hash(candidate_name))[:8]),
            "position": position,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "deception_prob": f"{dec_prob*100:.1f}",
            "risk_level": f"{risk_level} RISK",
            "confidence": f"{sum(data.get('confidence', [0.8]*5))/5 * 100:.0f}",
            "conf_status": "VALIDATED" if dec_prob < 0.7 else "UNSTABLE",
            "sync_score": f"{data.get('modality_confidence', {}).get('text', 0.5) * 100 + 40:.0f}", 
            "sync_status": "STABLE" if dec_prob < 0.5 else "FLUCTUATING",
            
            "visual_gaze_insight": "Slightly averted gaze during complex responses." if dec_prob > 0.5 else "Stable and direct gaze.",
            "visual_micro_insight": "Incongruent micro-expressions detected." if dec_prob > 0.5 else "Consistent emotional display.",
            "visual_presence_insight": "Professional and engaged posture.",
            
            "audio_prosody_insight": "Flattened pitch detected in critical segments." if dec_prob > 0.5 else "Natural pitch variation.",
            "audio_stress_insight": "Elevated vocal stress detected." if dec_prob > 0.5 else "Normal stress baseline.",
            "audio_confidence_insight": "Moderate vocal projection.",
            
            "text_complexity_insight": "High linguistic density.",
            "text_authenticity_insight": "Usage of hedging phrases noted." if dec_prob > 0.5 else "Direct and authentic language.",
            "text_topic_insight": "Strong adherence to the interview questions.",
            
            "o_level": o_lvl, "o_interpretation": o_interp,
            "c_level": c_lvl, "c_interpretation": c_interp,
            "e_level": e_lvl, "e_interpretation": e_interp,
            "a_level": a_lvl, "a_interpretation": a_interp,
            "n_level": n_lvl, "n_interpretation": n_interp,
            
            "incongruence_details": "Behavioral cues from audio (vocal stress) do not align with verbal statements." if dec_prob > 0.5 else "No significant behavioral incongruence detected.",
            
            "improvement_title_1": "Emotional Consistency" if dec_prob > 0.5 else "Leadership Presence",
            "improvement_desc_1": "Work on aligning non-verbal cues with verbal delivery to build trust." if dec_prob > 0.5 else "Focus on projecting authority in team settings.",
            "improvement_title_2": "Clarity of Response" if dec_prob > 0.5 else "Strategic Openness",
            "improvement_desc_2": "Minimize hedging language to appear more confident." if dec_prob > 0.5 else "Incorporate more diverse examples of problem-solving.",
            
            # Course placeholders — URLs shortened and focus trimmed to keep
            # training sequences under 1024 tokens. Real URLs live in edx_courses.json
            # and are used at inference time by generate_report.py.
            "course_title_1": rec_courses[0]["title"],
            "course_instructor_1": rec_courses[0]["instructor"],
            "course_focus_1": rec_courses[0]["focus"].split(".")[0][:80],
            "course_url_1": "https://edx.org",
            
            "course_title_2": rec_courses[1]["title"],
            "course_instructor_2": rec_courses[1]["instructor"],
            "course_focus_2": rec_courses[1]["focus"].split(".")[0][:80],
            "course_url_2": "https://edx.org",
            
            "short_term_advice": "Practice high-stakes communication drills." if dec_prob > 0.5 else "Lead a small cross-functional project.",
            "long_term_advice": "Develop a deep transparency communication style." if dec_prob > 0.5 else "Aim for senior strategic leadership roles."
        }
        
        # Fill template
        report = self.template
        for key, value in mapping.items():
            report = report.replace(f"{{{{{key}}}}}", str(value))
            
        return report

def main():
    BASE_DIR = Path(r"D:\Code\Grad\authentic_growth_reports")
    INPUT_DIR = BASE_DIR / "synthetic_dataset"
    TEMPLATE_PATH = BASE_DIR / "authentic_growth_template.md"
    OUTPUT_FILE = BASE_DIR / "finetuning_dataset.jsonl"
    
    gen = DatasetGenerator(TEMPLATE_PATH)
    
    files = sorted(list(INPUT_DIR.glob("synthetic_result_*.json")))
    print(f"Found {len(files)} files. Generating reports...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i, file_path in enumerate(files):
            with open(file_path, 'r', encoding='utf-8') as f_in:
                metrics = json.load(f_in)
            
            report_md = gen.process_metrics(metrics)
            
            model_input = {
                "traits": metrics["traits"],
                "deception_probability": metrics["deception_probability"],
                "confidence": metrics["confidence"],
                "modality_confidence": metrics["modality_confidence"]
            }
            
            record = {
                "instruction": "Generate an 'Authentic Growth' coaching report based on the following multimodal interview metrics.",
                "input": json.dumps(model_input),
                "output": report_md
            }
            
            f_out.write(json.dumps(record) + "\n")
            
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{len(files)} records generated.")
 
    print(f"Dataset complete! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
