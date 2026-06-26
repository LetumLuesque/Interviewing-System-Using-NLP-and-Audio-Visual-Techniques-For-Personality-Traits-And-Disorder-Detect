import json
import os
import re
from pathlib import Path

def sanitize_markdown(md_content):
    """Normalizes whitespace and ensures clean structure."""
    lines = [line.rstrip() for line in md_content.split('\n')]
    
    sanitized = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
        else:
            blank_count = 0
        if blank_count <= 1:
            sanitized.append(line)
            
    return '\n'.join(sanitized)

def validate_structure(md_content):
    """Checks if the Markdown contains all required template elements."""
    errors = []
    
    # 1. Check for unreplaced placeholders
    placeholders = re.findall(r'\{\{.*?\}\}', md_content)
    if placeholders:
        errors.append(f"Unreplaced placeholders found: {list(set(placeholders))}")
        
    # 2. Check for required headers
    required_headers = [
        "## Session Summary",
        "## Detailed Behavioral Insights",
        "## Working Style Profile",
        "## Coaching Roadmap"
    ]
    for header in required_headers:
        if header not in md_content:
            errors.append(f"Missing required header: {header}")
            
    # 3. Check for specific blockquotes/alerts
    required_blocks = [
        "Candidate Profile Summary",
        "Consistency Index:",
        "This roadmap outlines practical"
    ]
    for block in required_blocks:
        if block not in md_content:
            errors.append(f"Missing required blockquote section: {block}")
            
    # 4. Check for table integrity (crude check for pipe characters)
    if md_content.count('|') < 10:
        errors.append("Markdown tables seem incomplete or missing.")
        
    return errors

def main():
    BASE_DIR = Path(r"D:\Code\Grad\authentic_growth_reports")
    INPUT_FILE = BASE_DIR / "finetuning_dataset.jsonl"
    OUTPUT_FILE = BASE_DIR / "finetuning_dataset_sanitized.jsonl"
    REPORT_FILE = BASE_DIR / "validation_report.txt"
    
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Starting Sanitization Quality Loop on {INPUT_FILE}...")
    
    total_records = 0
    passed_count = 0
    failed_count = 0
    all_errors = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            total_records += 1
            data = json.loads(line)
            
            # Sanitization
            data['output'] = sanitize_markdown(data['output'])
            
            # Validation
            errors = validate_structure(data['output'])
            
            if not errors:
                f_out.write(json.dumps(data) + "\n")
                passed_count += 1
            else:
                failed_count += 1
                all_errors.append(f"Record {total_records}: {', '.join(errors)}")
            
            if total_records % 100 == 0:
                print(f"Processed {total_records} records...")

    # Write Report
    with open(REPORT_FILE, 'w', encoding='utf-8') as f_rep:
        f_rep.write(f"Validation Report - {INPUT_FILE.name}\n")
        f_rep.write("="*50 + "\n")
        f_rep.write(f"Total Records: {total_records}\n")
        f_rep.write(f"Passed: {passed_count}\n")
        f_rep.write(f"Failed: {failed_count}\n\n")
        
        if all_errors:
            f_rep.write("Errors Found:\n")
            for err in all_errors:
                f_rep.write(f"- {err}\n")
        else:
            f_rep.write("Result: 100% Quality Pass. Dataset is perfectly consistent.\n")

    print(f"Sanitization Complete!")
    print(f"Passed: {passed_count} | Failed: {failed_count}")
    print(f"Clean dataset saved to: {OUTPUT_FILE.name}")
    print(f"Full report saved to: {REPORT_FILE.name}")

if __name__ == "__main__":
    main()
