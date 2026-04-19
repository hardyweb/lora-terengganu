# =========================
# 1. INSTALL DEPENDENCIES
# =========================
!pip install -q unsloth datasets transformers accelerate trl bitsandbytes

# =========================
# 2. IMPORT LIBRARIES
# =========================
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# =========================
# 3. LOAD BASE MODEL
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-1B-Instruct",
    load_in_4bit = True,
    max_seq_length = 2048,
)

# =========================
# 4. APPLY LORA
# =========================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 16,
    lora_dropout = 0.05,
)

# =========================
# 5. DATASET (LOGHAT TERENGGANU)
# =========================
data = [
 # basic dialect usage
{"text": "Q: Kau gi mana tadi?\nA: Aku gi kedai je, beli mender sikit."},
{"text": "Q: Dah makan ke?\nA: Dok lagi, perut  bunyi doh ni."},
{"text": "Q: Kau buat apa tu?\nA: Aku tengoh lepak je, minum tea peng sambil nyembang."},

# slang real Terengganu words
{"text": "Q: Kenapa kau gelak?\nA: Sebab cerita tadi kelakor sangat."},
{"text": "Q: Kau nak ikut tak?\nA:Nok ahh, jom gi sekali."},
{"text": "Q: Kau lambat kenapa?\nA: Jalan jeng, aku pun jadi kalut sikit."},

# Pantai Timur natural speech
{"text": "Q: Kau penat ke?\nA: Penat weh, baru balik keje ni."},
{"text": "Q: Kau marah ke?\nA: Dok marah sangat, biasa je lah."},
{"text": "Q: Kau dengan sapa tadi?\nA: Dengan saim-saim je, duk sembang kosong."},

# strong dialect words
{"text": "Q: Kau ni macam malas je?\nA: Malah doh hari ni, nak rehat je."},
{"text": "Q: Kau lapar sangat ke?\nA: Haa lapor doh, perut bunyi dah ni."},
{"text": "Q: Kau nak balik ke?\nA: Nak balik doh, penat sangat dah."},

# cultural & identity
{"text": "Q: Orang Terengganu macam mana?\nA: Orang sini santai je, baik-baik belaka."},
{"text": "Q: Loghat ni susah ke?\nA: Mula-mula susah sikit, lama-lama biasa lah."},

# more dialect vocabulary style
{"text": "Q: Kau buat apa tadi?\nA: Aku gi kampung kejap, tengok pokok nyor."},
{"text": "Q: Kau suka kampung ke bandar?\nA: Kampung lagi molek doh, tenang je."},

# emotional tone Terengganu
{"text": "Q: Kau okay ke hari ni?\nA: Okay je, dok ada apa sangat pun."},
{"text": "Q: Kau sedih ke?\nA: Tak la sangat, hidup kena redho je."},

# food culture
{"text": "Q: Kau makan apa pagi ni?\nA: Nasi dagang lah, sedap molek kalau panas-panas."},
{"text": "Q: Kau suka makan apa?\nA: Aku suka ikang singgang, padu betul."},

# stronger slang expression
{"text": "Q: Kau ni bodoh ke?\nA: Eh tak lah, dok macam tu sangat."},
{"text": "Q: Kau nak gi mana malam ni?\nA: Nak gi lepak kedai kopi, sembang gelekek."},

# real Pantai Timur feel
{"text": "Q: Kau tak bosan ke duduk kampung?\nA: Dok lah, sini lagi tenang, hidup rilek-rilek je."},
{"text": "Q: Kau kenal dia ke?\nA: Kenal sikit-sikit je, dok rapat sangat pun."},

# reinforcement dialect mixing
{"text": "Q: Kenapa kau diam?\nA: tak dok nape, malas nak cakak banyak."},
{"text": "Q: Kau nak buat apa esok?\nA: Gi jalan-jalan je, tengok orang kampung."},
]

dataset = Dataset.from_list(data)

# =========================
# 6. TRAINING CONFIG
# =========================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        learning_rate = 2e-4,
        max_steps = 200,
        fp16 = True,
        logging_steps = 1,
        output_dir = "lora_terengganu",
    ),
)

# =========================
# 7. START TRAINING
# =========================
trainer.train()

# =========================
# 8. SAVE MODEL
# =========================
model.save_pretrained("lora_terengganu_adapter")
tokenizer.save_pretrained("lora_terengganu_adapter")

# =========================
# 9. TEST INFERENCE
# =========================
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    "Q: Apa itu AI?\nA:",
    return_tensors="pt"
).to("cuda")

output = model.generate(
    **inputs,
    max_new_tokens=120,
    temperature=0.8,
    top_p=0.9,
)

print(tokenizer.decode(output[0]))
