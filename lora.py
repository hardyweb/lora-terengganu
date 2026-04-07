# 1️⃣ Install dependencies
!pip install -q unsloth transformers accelerate peft trl datasets bitsandbytes

# 2️⃣ Import & load base model
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2b-bnb-4bit",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

# 3️⃣ Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    use_gradient_checkpointing=True,
)

# 4️⃣ Create a small Terengganu dialect dataset
import json
from datasets import Dataset

data = [
    {"instruction": "Tukar ayat ke loghat Terengganu", "input": "Apa khabar?", "output": "Gapo khaba?"},
    {"instruction": "Tukar ayat ke loghat Terengganu", "input": "Awak buat apa?", "output": "Mu buat gapo?"},
    {"instruction": "Jawab dalam loghat Terengganu", "input": "Saya lapar", "output": "Aku lapo doh"},
    {"instruction": "Balas dalam loghat Terengganu", "input": "Jom makan", "output": "Jom makang"},
]

dataset = Dataset.from_list(data)

# 5️⃣ Format prompt & remove original columns
def format_prompt(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    }

dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

# 6️⃣ Verify dataset
print(dataset[0]["text"])

# 7️⃣ Setup SFTTrainer
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=50,       # test cepat
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
    ),
)

# 8️⃣ Train
trainer.train()

# 9️⃣ Test output
prompt = """### Instruction:
Jawab dalam loghat Terengganu

### Input:
Saya nak pergi makan

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 10️⃣ Save model
model.save_pretrained("lora-terengganu")
tokenizer.save_pretrained("lora-terengganu")
