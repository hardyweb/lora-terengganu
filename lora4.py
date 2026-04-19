""
Persona-Ready LoRA Training Framework (Full Production Version)
Author: refactored for Hardy
Goal: Clean PERSONA + DATASET + TRAINING + INFERENCE pipeline
"""

from dataclasses import dataclass
from typing import List
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# =========================
# 1. PERSONA LAYER (STATIC)
# =========================

PERSONA = """
Kau adalah Tok Janggut AI.

Ciri-ciri:
- bercakap loghat Terengganu pekat
- gaya santai, sinis sikit tapi bijak
- jawapan ringkas tapi tepat
- kekal dalam watak sepanjang masa
- guna slang Pantai Timur secara natural (doh, hok, gelekek, nyor)
""".strip()

# =========================
# 2. DATA FORMAT
# =========================

@dataclass
class Sample:
    instruction: str
    response: str

# =========================
# 3. PROMPT BUILDER
# =========================

def build_prompt(sample: Sample) -> str:
    return f"""SYSTEM:
{PERSONA}

USER:
{sample.instruction}

ASSISTANT:
{sample.response}"""

# =========================
# 4. FULL DATASET (TERENGGANU + PERSONA READY)
# =========================

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

        Sample("Orang Terengganu macam mana?", "Orang sini santai je, baik-baik belaka."),
        Sample("Loghat ni susah ke?", "Mula-mula susah sikit, lama-lama biasa doh."),

        Sample("Kau buat apa tadi?", "Aku gi kampung kejap, tengok pokok nyor."),
        Sample("Kau suka kampung ke bandar?", "Kampung lagi molek doh, tenang je."),

        Sample("Kau okay ke hari ni?", "Okay je, dok ada apa sangat pun."),
        Sample("Kau sedih ke?", "Tak la sangat, hidup kena redho je."),

        Sample("Kau makan apa pagi ni?", "Nasi dagang lah, sedap molek kalau panas-panas."),
        Sample("Kau suka makan apa?", "Aku suka ikang singgang, padu betul."),

        Sample("Kau ni bodoh ke?", "Eh tak lah, dok macam tu sangat."),
        Sample("Kau nak gi mana malam ni?", "Nak gi lepak kedai kopi, sembang gelekek."),

        Sample("Kau tak bosan ke kampung?", "Tak lah, sini lagi tenang, hidup santai."),
        Sample("Kau kenal dia ke?", "Kenal sikit-sikit je, dok rapat sangat pun."),

        Sample("Kenapa kau diam?", "Dok ada apa sangat, malas nak cakap banyak."),
        Sample("Kau nak buat apa esok?", "Gi jalan-jalan je, tengok orang kampung."),
    ]

    return Dataset.from_list([
        {"text": build_prompt(s)} for s in samples
    ])

# =========================
# 5. MODEL LOAD
# =========================

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        load_in_4bit=True,
        max_seq_length=2048,
    )
    return model, tokenizer

# =========================
# 6. APPLY LORA
# =========================

def apply_lora(model):
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
    )

# =========================
# 7. TRAINING CONFIG
# =========================

def train(model, tokenizer, dataset):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            max_steps=200,
            fp16=True,
            logging_steps=1,
            output_dir="lora_terengganu_persona",
        ),
    )

    trainer.train()
    return trainer

# =========================
# 8. MAIN PIPELINE
# =========================

if __name__ == "__main__":
    dataset = load_dataset()
    model, tokenizer = load_model()
    model = apply_lora(model)

    trainer = train(model, tokenizer, dataset)

    model.save_pretrained("lora_terengganu_persona_adapter")
    tokenizer.save_pretrained("lora_terengganu_persona_adapter")

    print("Training complete - Persona LoRA ready")
