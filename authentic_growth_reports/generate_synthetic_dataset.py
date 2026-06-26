import json
import random
import os
from pathlib import Path

# Configuration
N = 1500
OUTPUT_DIR = Path(r"D:\Code\Grad\authentic_growth_reports\synthetic_dataset")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CANDIDATE_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Skyler", "Charlie", "Quinn"]
POSITIONS = ["Software Engineer", "Project Manager", "Data Analyst", "UX Designer", "Sales Representative", "HR Specialist", "Product Owner"]

def generate_synthetic_metrics():
    # 1. Personality Traits (OCEAN)
    # Most people are around the middle, so we use a mix of normal and uniform
    traits = [random.uniform(0.1, 0.9) for _ in range(5)]
    
    # 2. Deception Probability
    # We want a mix of honest (low prob) and deceptive (high prob) profiles
    # Using a beta distribution to get more samples at the extremes if needed, 
    # but uniform is fine for now to cover the whole range.
    deception_prob = random.random()
    
    # 3. Confidence Scores
    # Confidence usually correlates with the clarity of the signal
    base_conf = random.uniform(0.6, 0.95)
    confidence = [max(0.1, min(1.0, base_conf + random.uniform(-0.1, 0.1))) for _ in range(5)]
    
    # 4. Modality Contribution
    v = random.uniform(0.2, 0.5)
    a = random.uniform(0.2, 0.5)
    t = 1.0 - (v + a)
    # Ensure t is within bounds, if not, re-normalize
    if t < 0.1:
        total = v + a + 0.1
        v, a, t = v/total, a/total, 0.1/total
    
    modality_confidence = {
        "visual": round(v, 2),
        "audio": round(a, 2),
        "text": round(t, 2)
    }
    
    return {
        "traits": traits,
        "deception_probability": deception_prob,
        "confidence": confidence,
        "modality_confidence": modality_confidence,
        "metadata": {
            "candidate_name": random.choice(CANDIDATE_NAMES) + " " + str(random.randint(100, 999)),
            "position": random.choice(POSITIONS),
            "cultural_group": random.choice(["None", "English", "Arabic", "Chinese", "Spanish", "Hindi"])
        }
    }

def main():
    print(f"Generating {N} synthetic JSON objects...")
    
    dataset_summary = []
    
    for i in range(N):
        metrics = generate_synthetic_metrics()
        filename = f"synthetic_result_{i:04d}.json"
        filepath = OUTPUT_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
            
        dataset_summary.append({
            "id": i,
            "file": filename,
            "traits": metrics["traits"],
            "deception": metrics["deception_probability"]
        })
        
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{N} files created.")
            
    # Also save a master index for easier loading
    with open(OUTPUT_DIR / "dataset_index.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_summary, f, indent=4)
        
    print(f"Success! {N} files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
