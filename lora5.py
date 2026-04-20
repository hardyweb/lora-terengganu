# =====================================
# 0. INSTALL (COLAB SETUP)
# =====================================
!pip install -q unsloth transformers datasets accelerate trl xformers bitsandbytes

 # =====================================
# 1. IMPORT
# =====================================
from dataclasses import dataclass
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import torch

# =====================================
# 2. PERSONA
# =====================================
PERSONA = """
Kau adalah Tok Janggut AI.

Ciri-ciri:
- bercakap loghat Terengganu pekat
- gaya santai, sinis sikit tapi bijak
- jawapan ringkas tapi tepat
- kekal dalam watak sepanjang masa
- guna slang Pantai Timur secara natural (doh, hok, gelekek, nyor)
""".strip()


# =====================================
# 3. DATA STRUCTURE
# =====================================
@dataclass
class Sample:
    instruction: str
    response: str

def build_prompt(sample: Sample) -> str:
    return f"""SYSTEM:
{PERSONA}

USER:
{sample.instruction}

ASSISTANT:
{sample.response}"""


# =====================================
# 4. DATASET
# =====================================
def load_dataset():
    samples = [
        Sample("Kau gi mana tadi?", "Aku gi kedai je, beli mender sikit."),
        Sample("Dah makan ke?", "Belum lagi doh, perut dok bunyi ni."),
        Sample("Kau buat apa tu?", "Aku dok lepak je, minum peng sambil sembang."),
        Sample("Kenapa kau gelak?", "Sebab cerita tadi kelakar gelekek sangat."),
        Sample("Kau nak ikut tak?", "Nak doh, jom gi sekali."),
        Sample("Kau lambat kenapa?", "Jalan jem, aku pun jadi kelam kabut sikit."),
        Sample("Kau penat ke?", "Penat doh, baru balik keje."),
        Sample("Kau marah ke?", "Tak marah sangat, biasa je lah."),
        Sample("Kau dengan sapa tadi?", "Dengan kawan-kawan je, dok sembang kosong."),
        Sample("Kau ni malas ke?", "Malah doh hari ni, nak rehat je."),
        Sample("Kau lapar ke?", "Haa lapar doh, perut bunyi dah ni."),
        Sample("Kau nak balik ke?", "Nak balik doh, penat sangat dah."),
    ]

    return Dataset.from_list([{"text": build_prompt(s)} for s in samples])


# =====================================
# 5. LOAD MODEL (OPTIMIZED FOR T4)
# =====================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    load_in_4bit=True,
    max_seq_length=2048,
    dtype=None,  # auto detect fp16
)

FastLanguageModel.for_inference(model)  # speed boost


# =====================================
# 6. APPLY LORA
# =====================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,  # penting untuk T4
)


# =====================================
# 6. APPLY LORA
# =====================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,  # penting untuk T4
)

# =====================================
# 7. TRAINING
# =====================================
dataset = load_dataset()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,  # IMPORTANT for small dataset
    args=TrainingArguments(
        per_device_train_batch_size=2,   # T4 boleh handle
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=200,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="lora_output",
        report_to="none",
    ),
)

trainer.train()


# =====================================
# 8. SAVE MODEL
# =====================================
model.save_pretrained("lora_terengganu_persona_adapter")
tokenizer.save_pretrained("lora_terengganu_persona_adapter")

print("✅ Training complete")


# =====================================
# 9. QUICK INFERENCE TEST
# =====================================
FastLanguageModel.for_inference(model)

prompt = f"""SYSTEM:
{PERSONA}

USER:
Kau buat apa hari ni?

ASSISTANT:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

