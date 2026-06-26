import json
import random
from pathlib import Path
from generate_synthetic_dataset import generate_synthetic_metrics
from generate_training_pairs import DatasetGenerator

def main():
    BASE_DIR = Path(r"D:\Code\Grad\authentic_growth_reports")
    TEMPLATE_PATH = BASE_DIR / "authentic_growth_template.md"
    OUTPUT_FILE = BASE_DIR / "test_split.jsonl"
    
    # Initialize the report generator with the new template
    gen = DatasetGenerator(TEMPLATE_PATH)
    
    print("Generating 200 independent test split samples...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i in range(200):
            # Generate metrics using the same generator function
            metrics = generate_synthetic_metrics()
            
            # Ensure name indicates it's test set
            metrics["metadata"]["candidate_name"] = f"TestCandidate {i+1:03d}"
            
            # Process metrics using the same template generation logic
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
            
    print(f"Test split generation complete! Saved 200 records to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
