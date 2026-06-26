import os
os.environ["PYTHONUTF8"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1" # Disable Inductor compilation on Windows
import torch
from unsloth import FastVisionModel
import json
from pathlib import Path

# 1. Configuration
MODEL_PATH = r"D:\Code\Grad\authentic_growth_reports\gemma_4_weights"
LORA_PATH = r"D:\Code\Grad\authentic_growth_reports\gemma_4_e4b_lora"

def load_growth_model():
    """Loads the base model and the fine-tuned LoRA adapters."""
    print("🧠 Loading Authentic Growth Model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = MODEL_PATH,
        load_in_4bit = True,
        local_files_only = True,
        device_map = "cuda:0",
    )
    
    # Load LoRA adapters
    model = FastVisionModel.for_inference(model) # Optimized for inference
    model.load_adapter(LORA_PATH)
    
    return model, tokenizer

def generate_report(analysis_json_path, output_md_path):
    """Generates a Markdown report from a raw analysis JSON using the fine-tuned model."""
    model, tokenizer = load_growth_model()
    
    # Load analysis data
    with open(analysis_json_path, 'r') as f:
        data = json.load(f)
    
    # Format metrics in the exact schema that matches the dataset
    if "Openness" in data:
        traits = [
            data.get("Openness", 0.5),
            data.get("Conscientiousness", 0.5),
            data.get("Extraversion", 0.5),
            data.get("Agreeableness", 0.5),
            data.get("Neuroticism", 0.5)
        ]
        deception_probability = data.get("Deception_Prob", 0.0)
        confidence = [
            data.get("Video_Confidence", 0.8),
            data.get("Audio_Sincerity", 0.8),
            data.get("Text_Clarity", 0.8),
            0.8,
            0.8
        ]
        modality_confidence = {
            "visual": data.get("Video_Confidence", 0.8),
            "audio": data.get("Audio_Sincerity", 0.8),
            "text": data.get("Text_Clarity", 0.8)
        }
    else:
        traits = data.get("traits", [0, 0, 0, 0, 0])
        deception_probability = data.get("deception_probability", 0.0)
        confidence = data.get("confidence", [0.8, 0.8, 0.8, 0.8, 0.8])
        modality_confidence = data.get("modality_confidence", {})
        if not modality_confidence:
            modality_confidence = {
                "visual": confidence[0],
                "audio": confidence[1],
                "text": confidence[2]
            }

    # Normalize traits if needed (to match training data range of 0-1)
    if any(t > 1.1 for t in traits):
        traits = [t / 100.0 for t in traits]
        
    if deception_probability > 1.0:
        deception_probability = deception_probability / 100.0
        
    if any(c > 1.1 for c in confidence):
        confidence = [c / 100.0 for c in confidence]
        
    for key in list(modality_confidence.keys()):
        if modality_confidence[key] > 1.1:
            modality_confidence[key] = modality_confidence[key] / 100.0
                
    mapped_data = {
        "traits": traits,
        "deception_probability": deception_probability,
        "confidence": confidence,
        "modality_confidence": modality_confidence
    }
    
    # Format the input string (Metrics block)
    metrics_str = json.dumps(mapped_data, indent=2)
    
    prompt = f"### Instruction:\nGenerate an 'Authentic Growth' coaching report based on the following multimodal interview metrics.\n\n### Input:\n{metrics_str}\n\n### Response:\n"
    
    print("✍️ Generating Authentic Growth Report...")
    # Use processor correctly for Gemma 4 multimodal architecture
    inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
    
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            streamer = text_streamer,
            max_new_tokens = 1024,
            use_cache = True,
            temperature = 0.5,
            top_p = 0.9
        )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
    
    # Extract only the response part
    if "### Response:" in response:
        report_content = response.split("### Response:")[1].strip()
    else:
        report_content = response
        
    # Final cleanup: if the model loops and starts another report, cut it off
    header_marker = "# Authentic Growth:"
    if header_marker in report_content.split(header_marker, 1)[-1]:
        # There's a second header, cut it
        report_content = header_marker + report_content.split(header_marker)[1]

    # Force inject correct metadata into the AI's hallucinated template
    import re
    actual_name = data.get('candidate_name', 'Candidate')
    actual_pos = data.get('position', 'Applicant')
    actual_date = data.get('date', 'Today')
    
    report_content = re.sub(r"- Candidate Name:\s*`.*?`", f"- Candidate Name: `{actual_name}`", report_content)
    report_content = re.sub(r"- Target Position:\s*`.*?`", f"- Target Position: `{actual_pos}`", report_content)
    report_content = re.sub(r"- Analysis Date:\s*`.*?`", f"- Analysis Date: `{actual_date}`", report_content)
    report_content = re.sub(r"- Candidate ID:\s*`.*?`", f"- Candidate ID: `SYS-{abs(hash(actual_name)) % 100000:05d}`", report_content)
        
    # Ensure blank lines before lists so the markdown parser catches them
    report_content = re.sub(r'recommended:\s*\*', 'recommended:\n\n*', report_content)
    
    # Save the report
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # Generate HTML version
    html_path = str(output_md_path).replace(".md", ".html")
    generate_html_report(report_content, html_path)
    
    print(f"✅ Reports generated: {output_md_path} and {html_path}")
    return report_content
 
def generate_html_report(md_content, html_path):
    """Converts the AI Markdown into a premium HTML report with custom styling."""
    import markdown
    import re

    # 1. Convert Markdown to base HTML with extensions
    raw_html = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'sane_lists']
    )

    # 2. Transform the recommended edX courses list into a premium CSS Grid card component
    # Match the nested course list items carefully without matching across separate items
    course_pattern = r'<li>\s*<strong>([^<]+)</strong> \(([^)]+)\)\s*<ul>\s*<li>\s*<em>Focus:</em>\s*([^<]+)\s*</li>\s*<li>\s*<em>Link:</em>\s*<a href="([^"]+)">.*?</a>\s*</li>\s*</ul>\s*</li>'
    
    import json
    import urllib.parse
    from pathlib import Path
    
    course_url_map = {}
    try:
        courses_file = Path(__file__).parent / "edx_courses.json"
        with open(courses_file, 'r', encoding='utf-8') as f:
            for c in json.load(f):
                course_url_map[c['title'].strip().lower()] = c['url']
    except Exception:
        pass

    def replace_course(match):
        title = match.group(1).strip()
        instructor = match.group(2).strip()
        focus = match.group(3).strip()
        
        # Lookup real URL, fallback to an edX search query if the model invented the course
        url = course_url_map.get(title.lower())
        if not url:
            safe_query = urllib.parse.quote(title)
            url = f"https://www.edx.org/search?q={safe_query}"
        
        return f'''<div class="course-card">
        <span class="course-badge">edX Course</span>
        <div class="course-header">
            <h4 class="course-title">{title}</h4>
            <span class="course-instructor">{instructor}</span>
        </div>
        <p class="course-focus">{focus}</p>
        <a class="course-btn" href="{url}" target="_blank">Learn More &rarr;</a>
    </div>'''
    
    processed_html = re.sub(course_pattern, replace_course, raw_html)
    
    # Wrap the generated course cards into a clean grid, replacing the parent ul tags
    processed_html = re.sub(
        r'<ul>\s*(<div class="course-card">.*?</div>\s*<div class="course-card">.*?</div>)\s*</ul>',
        r'<div class="courses-grid">\1</div>',
        processed_html,
        flags=re.DOTALL
    )
    
    # Also handle single course list backup if any
    processed_html = re.sub(
        r'<ul>\s*(<div class="course-card">.*?</div>)\s*</ul>',
        r'<div class="courses-grid">\1</div>',
        processed_html,
        flags=re.DOTALL
    )

    styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=initial-scale=1.0">
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

        /* Header Branding */
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

        /* Headings */
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

        /* Blockquotes / Profile Card */
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

        /* Tables styling */
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

        /* Highlight classes */
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

        /* Lists */
        ul, ol {{
            margin-left: 20px;
            margin-bottom: 20px;
            font-size: 1.05rem;
        }}

        li {{
            margin-bottom: 10px;
        }}

        /* Recommended edX Course Cards */
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

        /* Print stylesheet */
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
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(styled_html)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Authentic Growth AI Report")
    parser.add_argument("--input", type=str, help="Path to analysis JSON")
    parser.add_argument("--output", type=str, help="Path to output Markdown")
    
    args = parser.parse_args()
    
    if args.input and args.output:
        generate_report(args.input, args.output)
    else:
        # Default test mode
        test_json = Path(r"D:\Code\Grad\authentic_growth_reports\sample_analysis.json")
        if not test_json.exists():
            sample_data = {
                "Openness": 0.82, "Conscientiousness": 0.91, "Extraversion": 0.45, 
                "Agreeableness": 0.78, "Neuroticism": 0.12, "Deception_Prob": 0.04,
                "Video_Confidence": 0.88, "Audio_Sincerity": 0.92, "Text_Clarity": 0.85
            }
            with open(test_json, 'w') as f:
                json.dump(sample_data, f)
        generate_report(test_json, r"D:\Code\Grad\authentic_growth_reports\test_ai_report.md")
