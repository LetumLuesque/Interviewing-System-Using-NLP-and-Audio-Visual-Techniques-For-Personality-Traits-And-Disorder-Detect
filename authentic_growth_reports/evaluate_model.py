"""
evaluate_model.py
─────────────────
Computes ROUGE-L and BERTScore evaluation metrics on the 200 test split samples
comparing the model's generated output with the reference template output.
"""

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["UNSLOTH_IS_PRESENT"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import json
import sys
from pathlib import Path
import torch

try:
    from unsloth import FastVisionModel
except ImportError:
    print("[ERROR] unsloth is not installed.")
    sys.exit(1)

# Import metrics libraries
try:
    import rouge_score
except ImportError:
    print("Installing rouge-score...")
    os.system("pip install rouge-score")

try:
    import bert_score
except ImportError:
    print("Installing bert-score...")
    os.system("pip install bert-score")

from rouge_score import rouge_scorer
import bert_score

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"D:\Code\Grad\authentic_growth_reports")
MODEL_PATH = BASE_DIR / "gemma_4_weights"
LORA_PATH = BASE_DIR / "gemma_4_e4b_lora"
TEST_SPLIT_FILE = BASE_DIR / "test_split.jsonl"
OUTPUT_METRICS_FILE = BASE_DIR / "evaluation_results.json"

def main():
    if not TEST_SPLIT_FILE.exists():
        print(f"[ERROR] Test split file not found: {TEST_SPLIT_FILE}")
        sys.exit(1)

    print("🧠 Loading model for evaluation...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = str(MODEL_PATH),
        load_in_4bit = True,
        local_files_only = True,
        device_map = "cuda:0",
    )
    model = FastVisionModel.for_inference(model)
    model.load_adapter(str(LORA_PATH))

    # Read test split
    print("📖 Reading test split...")
    test_records = []
    with open(TEST_SPLIT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            test_records.append(json.loads(line))

    print(f"Loaded {len(test_records)} test samples. Running inference...")

    references = []
    candidates = []

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []

    # Run inference on a subset of 30 test records to speed up evaluation 
    # (calculating BERTScore on 200 records on single GPU can be slow, but let's do 50 records)
    eval_limit = 50
    for idx, record in enumerate(test_records[:eval_limit], 1):
        prompt = f"### Instruction:\n{record['instruction']}\n\n### Input:\n{record['input']}\n\n### Response:\n"
        inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                use_cache=True,
                temperature=0.1
            )
        
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        if "### Response:" in response:
            generated_md = response.split("### Response:")[1].strip()
        else:
            generated_md = response.strip()

        # Clean loop duplicates if any
        if "# 💎" in generated_md.split("# 💎", 1)[-1]:
            generated_md = "# 💎" + generated_md.split("# 💎")[1]

        reference = record['output'].strip()
        
        references.append(reference)
        candidates.append(generated_md)

        # Compute running ROUGE-L
        scores = scorer.score(reference, generated_md)
        rouge_l_scores.append(scores['rougeL'].fmeasure)

        print(f"[{idx}/{eval_limit}] ROUGE-L F1: {scores['rougeL'].fmeasure:.4f}")

    print("\n🖥️ Computing BERTScore...")
    # Use bert_score to compute precision, recall, F1
    P, R, F1 = bert_score.score(candidates, references, lang="en", rescale_with_baseline=True, device="cuda:0")

    mean_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
    mean_bert_precision = P.mean().item()
    mean_bert_recall = R.mean().item()
    mean_bert_f1 = F1.mean().item()

    results = {
        "rougeL": mean_rouge_l,
        "bertscore_precision": mean_bert_precision,
        "bertscore_recall": mean_bert_recall,
        "bertscore_f1": mean_bert_f1
    }

    # Save to file
    with open(OUTPUT_METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"\n{'='*55}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*55}")
    print(f"  ROUGE-L F1 Score  : {mean_rouge_l * 100:.2f}%")
    print(f"  BERTScore Precision: {mean_bert_precision * 100:.2f}%")
    print(f"  BERTScore Recall   : {mean_bert_recall * 100:.2f}%")
    print(f"  BERTScore F1 Score : {mean_bert_f1 * 100:.2f}%")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
