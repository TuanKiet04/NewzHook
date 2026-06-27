import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ── CONFIG — chỉnh sửa phần này ─────────────────────────────────────────────
HF_CACHE        = "./models_cache"          # path tới cache vLLM đã pull
MODEL_ID        = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE       = "./pattern.jsonl"            # file dữ liệu của bạn
OUTPUT_DIR      = "./kietcorn-adapter"      # nơi lưu adapter sau khi train
NUM_EPOCHS      = 2                      
# ────────────────────────────────────────────────────────────────────────────

# Trỏ HuggingFace vào cache có sẵn, không download lại
os.environ["HF_HOME"] = HF_CACHE

# ── Load tokenizer ───────────────────────────────────────────────────────────
print("📦 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Load model với QLoRA (4-bit) để tiết kiệm VRAM trên T4 ──────────────────
print("🤖 Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,   # T4 dùng float16, không dùng bf16
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

# ── Load và format dataset ───────────────────────────────────────────────────
print("📂 Loading dataset...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def format_sample(sample):
    """Convert messages array → chuỗi text theo chat template Qwen"""
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_sample)
print(f"✅ Loaded {len(dataset)} samples")

# ── LoRA config ──────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=8,                       # rank — phù hợp với 50 mẫu
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[            # các layer cần train trong Qwen2.5
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    task_type="CAUSAL_LM",
    bias="none",
)

# ── Training config ──────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,      # effective batch size = 2 * 4 = 8
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,                 # chỉ giữ 2 checkpoint gần nhất
    bf16=False,                         # T4 không support bf16
    fp16=True,
    dataset_text_field="text",
    max_seq_length=4096,
    report_to="none",                   # tắt wandb
)

# ── Train ────────────────────────────────────────────────────────────────────
print("Start training...")
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=lora_config,
)

trainer.train()

# ── Save adapter (KHÔNG merge vào base model) ────────────────────────────────
print(f"Saving adapter to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done! Adapter saved to:", OUTPUT_DIR)
print("Adapter size:", end=" ")
os.system(f"du -sh {OUTPUT_DIR}")