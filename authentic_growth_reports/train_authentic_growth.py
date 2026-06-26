import os
os.environ["PYTHONUTF8"] = "1"
os.environ["UNSLOTH_IS_PRESENT"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_CE_LOSS_TARGET_GB"] = "4"

import torch
import torch.nn.functional as F
import sys

# 🛠️ DEEP MONKEY PATCH
def standard_ce_loss(logits, labels, **kwargs):
    # Standard PyTorch CrossEntropy logic
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

# Patch all possible namespaces
try:
    import unsloth_zoo.fused_losses.cross_entropy_loss as f1
    f1.unsloth_fused_ce_loss = standard_ce_loss
    
    # Also patch the zoo level if it exists
    import unsloth_zoo as zoo
    if hasattr(zoo, "unsloth_fused_ce_loss"):
        zoo.unsloth_fused_ce_loss = standard_ce_loss
        
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    print("[OK] Deep-patched Unsloth Fused Loss.")
except Exception as e:
    print(f"[INFO] Patch info: {e}")

from unsloth import FastVisionModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 1. Configuration Constants
MODEL_NAME = r"D:\Code\Grad\authentic_growth_reports\gemma_4_weights" 
DATASET_PATH = r"D:\Code\Grad\authentic_growth_reports\finetuning_dataset_sanitized.jsonl"
OUTPUT_DIR = r"D:\Code\Grad\authentic_growth_reports\gemma_4_e4b_lora"

def main():
    # 2. Initialize Gemma 4 Multimodal Model
    model, processor = FastVisionModel.from_pretrained(
        model_name = MODEL_NAME,
        load_in_4bit = True,
        use_gradient_checkpointing = True, 
        local_files_only = True,
        device_map = "cuda:0",
    )

    # 3. Apply QLoRA
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r = 16,
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_fused_cross_entropy = False,
    )

    # 4. Prepare Dataset
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input_data, output in zip(instructions, inputs, outputs):
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_data}\n\n### Response:\n{output}"
            texts.append(text)
        return { "text" : texts, }

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    def tokenize_function(examples):
        return processor.tokenizer(examples["text"], truncation=True, max_length=1152)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    model.config.use_cache = False

    # 5. Configure Standard Trainer
    trainer = Trainer(
        model = model,
        train_dataset = tokenized_dataset,
        data_collator = DataCollatorForLanguageModeling(processor.tokenizer, mlm=False),
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            num_train_epochs = 3,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            seed = 3407,
            output_dir = OUTPUT_DIR,
            report_to = "none",
            push_to_hub = False,
        ),
    )

    # 6. Execute Training
    print(f"Launching Deep-Patched Training for Gemma 4-E4B...")
    trainer.train()

    # 7. Save Results
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Success! LoRA adapters saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
