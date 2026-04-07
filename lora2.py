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

# Set EOS token
eos_token = tokenizer.eos_token

# 3️⃣ Apply LoRA - dengan lebih banyak target modules
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing=True,
)

# 4️⃣ Dataset yang DIBETULKAN - KONSISTEN menggunakan Kome/Deme
from datasets import Dataset

data = [
    # Ganti nama - konsisten "Kome" untuk Saya, "Deme" untuk Kamu
    {"instruction": "Tukar ayat ke loghat Terengganu", "input": "Apa khabar?", "output": "Gapo khabaq?"},
    {"instruction": "Tukar ayat ke loghat Terengganu", "input": "Awak buat apa?", "output": "Deme buat gapo?"},
    {"instruction": "Jawab dalam loghat Terengganu", "input": "Saya lapar", "output": "Kome lapo doh"},
    {"instruction": "Balas dalam loghat Terengganu", "input": "Jom makan", "output": "Jom makang"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak pergi makan", "output": "Kome nok gi makang"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia tidak tahu", "output": "Dia dok tau"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya baik-baik saja", "output": "Kome bereh je"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "deme", "output": "Kamu / awak (kata ganti nama untuk orang kedua)"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kome", "output": "Saya / aku (kata ganti nama untuk orang pertama)"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "nok", "output": "Nak / mahu / hendak"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "gi", "output": "Pergi"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "khabaq", "output": "Khabar / berita"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu mahu ke mana?", "output": "Deme nok gi mane?"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya mahu pulang", "output": "Kome nok balek"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Sudah makan belum?", "output": "Makan doh?"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Sudah makan", "output": "Makan doh"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Belum makan", "output": "Dok makang lagik"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia sangat cantik", "output": "Dia comel sangat"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Rumah itu besar", "output": "Rumoh tu besor"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Air sungai itu jernih", "output": "Aie sungoi tu jenih"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya tidak faham", "output": "Kome dok paham"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu faham tak?", "output": "Deme paham dak?"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "balek", "output": "Balik / pulang"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "besor", "output": "Besar"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kecik", "output": "Kecil"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "mane", "output": "Mana"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "gapo", "output": "Apa"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "sape", "output": "Siapa"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Siapa nama kamu?", "output": "Sape namo deme?"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Nama saya Ali", "output": "Namo kome Ali"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Hari ini panas", "output": "Ari ni pane"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Hari ini hujan", "output": "Ari ni ujan"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "ari", "output": "Hari"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "pane", "output": "Panas"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "ujan", "output": "Hujan"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya orang Terengganu", "output": "Kome oghe Ganu"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Mari sini", "output": "Mai sini"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pergi sana", "output": "Gi sano"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Makan nasi lemak", "output": "Makang nasi lemok"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Tolong bantu saya", "output": "Tolong bantu kome"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Terima kasih", "output": "Terimo kasih"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Sama-sama", "output": "Samo-samo"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "oghe", "output": "Orang"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "makang", "output": "Makan"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia sedang tidur", "output": "Dia tengah tiduq"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya suka awak", "output": "Kome suko deme"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu sangat baik", "output": "Deme baik sangat"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Jangan marah", "output": "Jangan mareh"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya lapar", "output": "Kome lapo"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia sudah tidur", "output": "Dah tiduq doh"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Berapa harga ini?", "output": "Brapo hargo ni?"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Murah sikit", "output": "Murah siket"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "brapo", "output": "Berapa"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "hargo", "output": "Harga"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "siket", "output": "Sikit / sedikit"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak minum air", "output": "Kome nok minum aie"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu sudah mandi?", "output": "Deme dah mandi?"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya sudah mandi", "output": "Kome dah mandi"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kerja apa kamu buat?", "output": "Kejo gapo deme buak?"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya kerja pejabat", "output": "Kome kejo opis"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kejo", "output": "Kerja"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "buak", "output": "Buat"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "aie", "output": "Air"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Esok saya balik kampung", "output": "Esok kome balek kampung"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Semalam saya datang lambat", "output": "Semalam kome datang lambak"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "lambak", "output": "Lambat"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ikan itu segar", "output": "Ikan tu segor"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Laut itu indah", "output": "Lauk tu indah"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "lauk", "output": "Laut / Ikan goreng"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak beli ikan", "output": "Kome nok beli ikan"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu pergi pasar?", "output": "Deme gi paso?"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ya, saya pergi pasar", "output": "Ho, kome gi paso"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "paso", "output": "Pasar"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Tidak, saya tidak pergi", "output": "Dok, kome dok gi"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia pandai masak", "output": "Dia pandai masak"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya tidak pandai masak", "output": "Kome dok pandai masak"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Rumah saya dekat pantai", "output": "Rumoh kome dokat pantai"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "dokat", "output": "Dekat / berhampiran"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya letih hari ini", "output": "Kome letih ari ni"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu rehat dulu", "output": "Deme rehat dulu"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ayam itu gemuk", "output": "Ayam tu gemok"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kucing itu kurus", "output": "Kucing tu kuruih"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "gemok", "output": "Gemuk"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kuruih", "output": "Kurus"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya gembira hari ini", "output": "Kome gumbiro ari ni"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia sedih sangat", "output": "Dia sedih sangat"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya sakit kepala", "output": "Kome sakik kapalo"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pergi jumpa doktor", "output": "Gi jumpe dokto"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kapalo", "output": "Kepala"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu mahu makan apa?", "output": "Deme nok makang gapo?"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya mahu makan nasi goreng", "output": "Kome nok makang nasi goring"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Tolong jangan bising", "output": "Tolong jangan bising"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak tidur awal", "output": "Kome nok tiduq awal"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Bangun pagi esok", "output": "Bangun pagi esok"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia lambat datang", "output": "Dia lambak datang"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya sudah siap", "output": "Kome dah siap"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu belum siap lagi", "output": "Deme belom siap lagik"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Cepat sikit", "output": "Cepat siket"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Perlahan sikit", "output": "Pelan siket"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "pelan", "output": "Perlahan"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya suka makan ikan bakar", "output": "Kome suko makang ikan bakaq"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "bakaq", "output": "Bakar"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu sudah makan tengah hari?", "output": "Deme dah makang tengah ari?"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "tengah ari", "output": "Tengah hari"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak minum petang", "output": "Kome nok minum petang"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Malam ini gelap", "output": "Malam ni gelap"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya sayang kamu", "output": "Kome sayang deme"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Keluarga saya besar", "output": "Kaluargo kome besor"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kaluargo", "output": "Keluarga"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ayah saya pergi kerja", "output": "Ayah kome gi kejo"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ibu saya masak di dapur", "output": "Mak kome masak di dapo"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "dapo", "output": "Dapur"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Adik saya pergi sekolah", "output": "Adik kome gi sekolah"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Jangan malas", "output": "Jangan malas"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak pergi kedai", "output": "Kome nok gi kedai"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Tolong tunggu saya", "output": "Tolong tunggu kome"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya sudah sampai", "output": "Kome dah sampai"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu di mana?", "output": "Deme di mane?"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya di rumah", "output": "Kome di rumoh"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kita pergi bersama", "output": "Kite gi samo-samo"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya tidak mahu pergi", "output": "Kome dok nok gi"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia mahu ikut", "output": "Dia nok ikut"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Hari ini hari Jumaat", "output": "Ari ni ari Jumaat"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak tidur tengah hari", "output": "Kome nok tiduq tengah ari"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Cuaca cantik hari ini", "output": "Cuaca cantik ari ni"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Awan gelap di langit", "output": "Awan gelap di langit"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak pergi memancing", "output": "Kome nok gi mancing"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pancing ikan di sungai", "output": "Mancing ikan di sungoi"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "sungoi", "output": "Sungai"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ikan besar di laut", "output": "Ikan besor di lauk"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya suka pantai", "output": "Kome suko pantai"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pasir putih bersih", "output": "Pasi puteh bersih"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "pasi", "output": "Pasir"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "puteh", "output": "Putih"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Terengganu negeri saya", "output": "Ganu negeri kome"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya bangga jadi orang Terengganu", "output": "Kome banggo jadi oghe Ganu"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "banggo", "output": "Bangga"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Loghat Terengganu unik", "output": "Loghat Ganu unik"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya belajar loghat Terengganu", "output": "Kome belajar loghat Ganu"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu faham loghat Terengganu?", "output": "Deme paham loghat Ganu dak?"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Sikit-sikit saya faham", "output": "Siket-siket kome paham"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia fasih loghat Terengganu", "output": "Dia fasih loghat Ganu"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak belajar lagi", "output": "Kome nok belajar lagik"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu ajar saya", "output": "Deme ajar kome"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Terima kasih banyak", "output": "Terimo kasih banyak"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Sama-sama, jangan lupa", "output": "Samo-samo, jangan lupo"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "lupo", "output": "Lupa"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya tidak akan lupa", "output": "Kome dok kan lupo"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kita jumpa lagi esok", "output": "Kite jumpe lagik esok"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Selamat tinggal", "output": "Selamat tinggal"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya gembira dapat kenal kamu", "output": "Kome gumbiro dapat kenal deme"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu kawan saya", "output": "Deme kawan kome"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kawan", "output": "Kawan / teman"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya rindu kamu", "output": "Kome rindu deme"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya rindu masakan ibu", "output": "Kome rindu masakan mak"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Masakan Terengganu sedap", "output": "Masakan Ganu sedap"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Nasi dagang paling sedap", "output": "Nasi dagang paling sedap"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Keropok lekor makanan kegemaran", "output": "Keropok lekor maknan kegemaran"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya suka keropok lekor", "output": "Kome suko keropok lekor"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Keropok itu rangup", "output": "Keropok tu rangup"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak pergi Kuala Terengganu", "output": "Kome nok gi Kuala Ganu"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pasar Payang di Kuala Terengganu", "output": "Paso Payang di Kuala Ganu"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pantai Batu Buruk cantik", "output": "Pantai Batu Buruk cantik"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pulau Perhentian indah", "output": "Pulau Perhentian indah"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Air laut jernih dan biru", "output": "Aie lauk jenih dan biru"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya takut ikan besar", "output": "Kome takut ikan besor"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Jangan takut, ikan kecil saja", "output": "Jangan takut, ikan kecik je"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Penyu ada di sini", "output": "Penyu ado di sini"},
    {"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "ado", "output": "Ada"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pulau Redang juga cantik", "output": "Pulau Redang jugo cantik"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya suka semua pulau", "output": "Kome suko sumo pulau"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Terengganu negeri paling cantik", "output": "Ganu negeri paling cantik"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak duduk sini selamanya", "output": "Kome nok duduk sini selamo-ne"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Hidup orang Terengganu", "output": "Hidup oghe Ganu"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya sayang Terengganu", "output": "Kome sayang Ganu"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Terengganu maju dan pesat", "output": "Ganu maju dan pesat"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kita semua bersatu", "output": "Kite sumo bersatu"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kita kuat bersama", "output": "Kite kuat samo-samo"},
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Bersama kita boleh", "output": "Samo-samo kite boleh"},
]

dataset = Dataset.from_list(data)

# 5️⃣ Format prompt DENGAN EOS TOKEN
def format_prompt(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}{eos_token}"""
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
        num_train_epochs=3,  # Ganti max_steps dengan epochs untuk training sebenar
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        output_dir="outputs",
        save_strategy="epoch",
    ),
)

# 8️⃣ Train
trainer.train()

# 9️⃣ Test output - GUNA FastLanguageModel untuk inference yang lebih baik
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

prompt = """### Instruction:
Jawab dalam loghat Terengganu

### Input:
Saya nak pergi makan

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.3,  # Lower temperature untuk lebih konsisten
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
# Extract only the response part
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response.split("### Response:")[-1].strip()
print("Jawapan:", response)

# 🔟 Save model
model.save_pretrained("lora-terengganu")
tokenizer.save_pretrained("lora-terengganu")

# Opsional: Merge dan save full model
# model.save_pretrained_merged("gemma-terengganu-full", tokenizer, save_method="merged_16bit")
