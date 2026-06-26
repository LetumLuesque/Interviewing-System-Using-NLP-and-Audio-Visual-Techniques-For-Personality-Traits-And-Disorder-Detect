
import os
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

def save_uploaded_file(uploaded_file, save_dir: str) -> str:
    """
    Saves an uploaded file to the specified directory.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def run_inference_script(
    script_path: str,
    checkpoint_path: str,
    video_path: str,
    output_dir: str,
    cwd: str,
    candidate_name: str = "Candidate",
    position: str = "Applicant",
    fusion_method: str = "concat",
    additional_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Runs the inference script as a subprocess and captures output.
    """
    cmd = [
        "python", script_path,
        "--checkpoint", checkpoint_path,
        "--video", video_path,
        "--output_dir", output_dir,
        "--candidate_name", candidate_name,
        "--position", position,
        "--fusion_method", fusion_method.lower()
    ]
    
    if additional_args:
        for key, value in additional_args.items():
            if value is True:
                cmd.append(f"--{key}")
            elif value is not False and value is not None:
                cmd.extend([f"--{key}", str(value)])

    print(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        universal_newlines=True
    )
    
    return process

def parse_json_result(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Looks for the analysis_results.json file in the output directory.
    """
    json_path = os.path.join(output_dir, "analysis_results.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON result: {e}")
            return None
    return None

def update_scoreboard(candidate_name: str, position: str, results: Dict[str, Any], scoreboard_path: str):
    """
    Updates the scoreboard JSON file with new candidate results.
    """
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": candidate_name,
        "position": position,
        "traits": results.get("traits", [0,0,0,0,0]),
        "deception": results.get("deception_probability", 0.0)
    }
    
    scoreboard = []
    if os.path.exists(scoreboard_path):
        try:
            with open(scoreboard_path, 'r') as f:
                scoreboard = json.load(f)
        except:
            scoreboard = []
            
    scoreboard.append(entry)
    
    with open(scoreboard_path, 'w') as f:
        json.dump(scoreboard, f, indent=4)

def get_scoreboard(scoreboard_path: str) -> list:
    """
    Retrieves the list of all scores from the scoreboard.
    """
    if os.path.exists(scoreboard_path):
        try:
            with open(scoreboard_path, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def run_gemma_growth_report(
    analysis_json_path: str,
    output_md_path: str,
    inference_script_path: str = r"D:\Code\Grad\authentic_growth_reports\inference_growth_report.py"
) -> subprocess.Popen:
    """
    Runs the Gemma 4 fine-tuned model to generate an Authentic Growth report.
    Enforces UTF8 mode and stops looping.
    """
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["TORCH_COMPILE_DISABLE"] = "1"
    
    cmd = [
        "python", inference_script_path,
        "--input", analysis_json_path,
        "--output", output_md_path
    ]
    
    # We'll modify the inference script slightly to take CLI args if needed, 
    # but for now it reads test_json. Let's make it more flexible.
    
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        universal_newlines=True
    )
    
    return process

def generate_demo_growth_report(
    analysis_json_path: str,
    output_md_path: str,
    output_html_path: str,
    candidate_name: str = "N/A",
    position: str = "N/A"
) -> bool:
    """
    Generates the Authentic Growth coaching report (.md and .html) instantly for DEMO_MODE.
    Avoids loading PyTorch/Unsloth models.
    """
    import json
    import re
    from datetime import datetime
    from pathlib import Path
    import markdown

    try:
        # 1. Load template
        template_path = Path(r"D:\Code\Grad\authentic_growth_reports\authentic_growth_template.md")
        if not template_path.exists():
            return False
            
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        # 2. Load edX courses
        courses_file = Path(r"D:\Code\Grad\authentic_growth_reports\edx_courses.json")
        if not courses_file.exists():
            return False
            
        with open(courses_file, 'r', encoding='utf-8') as f:
            all_courses = json.load(f)

        courses_by_cat = {}
        for c in all_courses:
            courses_by_cat.setdefault(c["category"], []).append(c)

        # 3. Load analysis data
        with open(analysis_json_path, 'r') as f:
            data = json.load(f)

        traits = data.get('traits', [0]*5)
        # Normalize if needed
        if max(traits) <= 1.0:
            traits = [t * 100 for t in traits]

        dec_prob = data.get('deception_probability', 0)
        risk_level = "HIGH" if dec_prob > 0.6 else "MEDIUM" if dec_prob > 0.3 else "LOW"

        def _interpret_trait(score, low_desc, mod_desc, high_desc):
            if score < 30:
                return "Low", low_desc
            elif score < 70:
                return "Moderate", mod_desc
            else:
                return "High", high_desc

        o_lvl, o_interp = _interpret_trait(
            traits[0], 
            "Prefers routine and familiar situations over new approaches", 
            "Balanced between traditional methods and open-minded innovation", 
            "Highly creative, imaginative, and receptive to new ideas"
        )
        c_lvl, c_interp = _interpret_trait(
            traits[1], 
            "Prefers a flexible, spontaneous approach to tasks rather than rigid planning", 
            "Demonstrates a balanced approach to task organization and schedules", 
            "Highly organized, methodical, and strongly goal-oriented"
        )
        e_lvl, e_interp = _interpret_trait(
            traits[2], 
            "Tends to work best in quiet, independent environments", 
            "Comfortable in both social settings and quiet individual workspaces", 
            "Energized by collaborative activities and social interactions"
        )
        a_lvl, a_interp = _interpret_trait(
            traits[3], 
            "Values directness and competitive drive in communication", 
            "Balances personal assertiveness with team collaboration", 
            "Highly cooperative, supportive, and empathetic toward team members"
        )
        n_lvl, n_interp = _interpret_trait(
            traits[4], 
            "Highly stable, calm, and resilient under high pressure", 
            "Maintains a typical, balanced response to workplace stress", 
            "Sensitive to high-pressure environments; thrives with extra support"
        )

        # Select courses
        needs = []
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
            
        needs.append(("leadership", 10))
        needs.append(("strategy", 5))
        
        needs.sort(key=lambda x: x[1], reverse=True)
        
        selected_courses = []
        selected_cats = set()
        
        for cat, delta in needs:
            if cat in courses_by_cat and cat not in selected_cats:
                possible = courses_by_cat[cat]
                idx = int(sum(traits)) % len(possible)
                selected_courses.append(possible[idx])
                selected_cats.add(cat)
            if len(selected_courses) == 2:
                break
                
        while len(selected_courses) < 2:
            for cat in courses_by_cat:
                if cat not in selected_cats:
                    selected_courses.append(courses_by_cat[cat][0])
                    selected_cats.add(cat)
                    break
                    
        rec_courses = selected_courses

        # Mapping
        mapping = {
            "candidate_name": candidate_name,
            "candidate_id": Path(analysis_json_path).parent.name if Path(analysis_json_path).parent.name else "CAND_ID",
            "position": position,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "deception_prob": f"{dec_prob*100:.1f}",
            "risk_level": f"{risk_level} RISK",
            "confidence": "85", 
            "conf_status": "VALIDATED",
            "sync_score": "92",
            "sync_status": "STABLE",
            
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
            
            "course_title_1": rec_courses[0]["title"],
            "course_instructor_1": rec_courses[0]["instructor"],
            "course_focus_1": rec_courses[0]["focus"],
            "course_url_1": rec_courses[0]["url"],
            
            "course_title_2": rec_courses[1]["title"],
            "course_instructor_2": rec_courses[1]["instructor"],
            "course_focus_2": rec_courses[1]["focus"],
            "course_url_2": rec_courses[1]["url"],
            
            "short_term_advice": "Practice high-stakes communication drills." if dec_prob > 0.5 else "Lead a small cross-functional project.",
            "long_term_advice": "Develop a deep transparency communication style." if dec_prob > 0.5 else "Aim for senior strategic leadership roles."
        }

        # Fill template
        report_content = template
        for key, value in mapping.items():
            report_content = report_content.replace(f"{{{{{key}}}}}", str(value))

        # Save Markdown report
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # 4. Generate HTML report
        raw_html = markdown.markdown(
            report_content,
            extensions=['tables', 'fenced_code', 'sane_lists']
        )

        # Transform recommended edX courses list into premium cards
        course_pattern = r'<li>\s*<strong>(.*?)</strong> \((.*?)\)\s*<ul>\s*<li>\s*<em>Focus:</em>\s*(.*?)\s*</li>\s*<li>\s*<em>Link:</em>\s*<a href="(.*?)">.*?</a>\s*</li>\s*</ul>\s*</li>'
        course_card_replacement = r'''<div class="course-card">
            <span class="course-badge">edX Course</span>
            <div class="course-header">
                <h4 class="course-title">\1</h4>
                <span class="course-instructor">\2</span>
            </div>
            <p class="course-focus">\3</p>
            <a class="course-btn" href="\4" target="_blank">Learn More &rarr;</a>
        </div>'''
        
        processed_html = re.sub(course_pattern, course_card_replacement, raw_html, flags=re.DOTALL)
        processed_html = re.sub(
            r'<ul>\s*(<div class="course-card">.*?</div>\s*<div class="course-card">.*?</div>)\s*</ul>',
            r'<div class="courses-grid">\1</div>',
            processed_html,
            flags=re.DOTALL
        )
        processed_html = re.sub(
            r'<ul>\s*(<div class="course-card">.*?</div>)\s*</ul>',
            r'<div class="courses-grid">\1</div>',
            processed_html,
            flags=re.DOTALL
        )

        # Use the premium styling
        styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Authentic Growth Executive Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #F8F6F1;
            --card-bg: #FFFFFF;
            --text-color: #2D3748;
            --text-title: #1B2A4A;
            --accent-gold: #C9A84C;
            --accent-gold-hover: #B0923B;
            --status-green: #2E7D62;
            --status-red: #A63228;
            --border-color: #E2E8F0;
            --shadow-sm: 0 4px 6px rgba(27, 42, 74, 0.03);
            --shadow-md: 0 10px 15px -3px rgba(27, 42, 74, 0.05), 0 4px 6px -2px rgba(27, 42, 74, 0.02);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 50px 20px;
        }}

        .report-wrapper {{
            max-width: 900px;
            margin: 0 auto;
            background-color: var(--card-bg);
            padding: 50px 60px;
            border-radius: 16px;
            box-shadow: var(--shadow-md);
            border-top: 6px solid var(--accent-gold);
        }}

        .report-meta-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--bg-color);
            padding-bottom: 15px;
            margin-bottom: 35px;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--accent-gold);
            font-weight: 600;
        }}

        h1 {{
            font-family: 'Playfair Display', serif;
            font-size: 2.4rem;
            color: var(--text-title);
            line-height: 1.2;
            margin-bottom: 25px;
            font-weight: 700;
        }}

        h2 {{
            font-family: 'Playfair Display', serif;
            font-size: 1.6rem;
            color: var(--text-title);
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
        }}

        h3 {{
            font-size: 1.15rem;
            color: var(--text-title);
            margin-top: 25px;
            margin-bottom: 12px;
            font-weight: 600;
        }}

        p {{
            margin-bottom: 18px;
            font-size: 1.05rem;
            color: var(--text-color);
        }}

        blockquote {{
            background-color: #FAF9F6;
            border-left: 3px solid var(--accent-gold);
            padding: 20px 25px;
            margin: 25px 0;
            border-radius: 0 8px 8px 0;
            box-shadow: var(--shadow-sm);
        }}

        blockquote p {{
            margin-bottom: 0;
            font-size: 1rem;
        }}

        blockquote ul {{
            list-style: none;
            margin-top: 10px;
        }}

        blockquote li {{
            margin-bottom: 8px;
            font-size: 0.95rem;
        }}

        blockquote strong {{
            color: var(--text-title);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.95rem;
            box-shadow: var(--shadow-sm);
            border-radius: 8px;
            overflow: hidden;
        }}

        th {{
            background-color: var(--text-title);
            color: #FFFFFF;
            text-align: left;
            padding: 14px 18px;
            font-weight: 500;
            letter-spacing: 0.02em;
        }}

        td {{
            padding: 14px 18px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-color);
        }}

        tr:nth-child(even) td {{
            background-color: #FAF9F6;
        }}

        tr:last-child td {{
            border-bottom: 2px solid var(--border-color);
        }}

        strong {{
            font-weight: 600;
        }}

        code {{
            background-color: #F1EFF0;
            color: var(--text-title);
            font-family: inherit;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 500;
            font-size: 0.95rem;
        }}

        ul, ol {{
            margin-left: 20px;
            margin-bottom: 20px;
            font-size: 1.05rem;
        }}

        li {{
            margin-bottom: 10px;
        }}

        .courses-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 25px 0;
        }}

        .course-card {{
            background-color: #FFFFFF;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 22px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
            box-shadow: var(--shadow-sm);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .course-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(27, 42, 74, 0.08);
            border-color: var(--accent-gold);
        }}

        .course-badge {{
            position: absolute;
            top: 15px;
            right: 15px;
            background-color: rgba(201, 168, 76, 0.1);
            color: var(--accent-gold-hover);
            font-size: 0.75rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 30px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .course-header {{
            margin-bottom: 12px;
            padding-right: 65px;
        }}

        .course-title {{
            font-size: 1.05rem;
            color: var(--text-title);
            font-weight: 600;
            line-height: 1.3;
            margin-bottom: 4px;
        }}

        .course-instructor {{
            font-size: 0.85rem;
            color: #718096;
            font-weight: 500;
        }}

        .course-focus {{
            font-size: 0.92rem;
            color: var(--text-color);
            margin-bottom: 20px;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}

        .course-btn {{
            display: inline-block;
            text-decoration: none;
            color: #FFFFFF;
            background-color: var(--text-title);
            font-size: 0.88rem;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 6px;
            text-align: center;
            transition: background-color 0.2s ease;
        }}

        .course-btn:hover {{
            background-color: var(--accent-gold);
        }}

        @media print {{
            body {{
                background-color: #FFFFFF;
                color: #000000;
                padding: 0;
            }}

            .report-wrapper {{
                box-shadow: none;
                padding: 0;
                border-top: none;
            }}

            .course-card {{
                page-break-inside: avoid;
                box-shadow: none;
                border: 1px solid #CBD5E0;
            }}
        }}

        @media (max-width: 768px) {{
            .report-wrapper {{
                padding: 30px 20px;
            }}
            .courses-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-wrapper">
        <div class="report-meta-header">
            <span>Executive Assessment Dossier</span>
            <span>Authentic Growth v3.0</span>
        </div>
        {processed_html}
    </div>
</body>
</html>"""

        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(styled_html)

        return True
    except Exception as e:
        print(f"Error generating demo growth report: {e}")
        return False

