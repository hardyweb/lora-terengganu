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
    {"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak pergi makan", "output": "Amber nok gi makang"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia tidak tahu", "output": "Dia dok tau"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Apa khabar?", "output": "Guane gamok?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya baik-baik saja", "output": "Amber bereh je"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "Tak apa", "output": "Dok apa / tidak mengapa"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "deme", "output": "Kamu / awak (kata ganti nama untuk orang kedua)"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "Ambe", "output": "Saya / aku (kata ganti nama untuk orang pertama)"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "nok", "output": "Nak / mahu / hendak"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "gi", "output": "Pergi"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "habo", "output": "Khabar / berita"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu mahu ke mana?", "output": "Deme nok gi mane?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya mahu pulang", "output": "Ambe nok balek"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Sudah makan belum?", "output": "Makan doh?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Sudah makan", "output": "Makan doh"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Belum makan", "output": "Dok makang lagi"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia sangat cantik", "output": "Dia comel sangat"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Rumah itu besar", "output": "Bapok besor rumoh tu"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Air sungai itu jernih", "output": "Air sungai tu jenih"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya tidak faham", "output": "amber dok paham"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu faham tak?", "output": "Deme paham dak?"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "balek", "output": "Balik / pulang"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "beso", "output": "Besar"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kecik", "output": "Kecil"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "mano", "output": "Mana"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "ape", "output": "Apa"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "sape", "output": "Siapa"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Siapa nama kamu?", "output": "Sape namo demo?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Nama saya Ali", "output": "Namo kome Ali"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Hari ini panas", "output": "Ari ni pane"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Hari ini hujan", "output": "Ari ni ujan"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "ari", "output": "Hari"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "pane", "output": "Panas"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "ujan", "output": "Hujan"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "nighi", "output": "Negeri"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "ganu", "output": "Terengganu (nama singkatan)"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya orang Terengganu", "output": "Kome oghe Ganu"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Mari sini", "output": "Mai sini"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pergi sana", "output": "Gi sano"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "mai", "output": "Mari"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "sano", "output": "Sana"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Makan nasi lemak", "output": "Makang nasi lemok"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Tolong bantu saya", "output": "Tolong bantu kome"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Terima kasih", "output": "Teimo kasih"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Sama-sama", "output": "Samo-samo"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "teime", "output": "Terima"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "sama", "output": "Sama"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "oghang", "output": "Orang"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "makang", "output": "Makan"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia sedang tidur", "output": "Dia tgh tidur"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya suka awak", "output": "Kome suko demo"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu sangat baik", "output": "Demo baik sangat"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Jangan marah", "output": "Jangan mrah"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya lapar", "output": "Kome lapo"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia sudah tidur", "output": "Dia dah tiduq"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "lapo", "output": "Lapar"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "tiduq", "output": "Tidur"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "suko", "output": "Suka"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "mrah", "output": "Marah"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Berapa harga ini?", "output": "Brapo hargo ni?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Mahal sangat", "output": "Mahal sangat"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Murah sikit", "output": "Murah siket"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "brapo", "output": "Berapa"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "hargo", "output": "Harga"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "siket", "output": "Sikit / sedikit"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak minum air", "output": "Kome nok minum aie"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu sudah mandi?", "output": "Demo dah mandi?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya sudah mandi", "output": "Kome dah mandi"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kerja apa kamu buat?", "output": "Kejo apo demo buak?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya kerja pejabat", "output": "Kome kejo opis"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kejo", "output": "Kerja"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "buak", "output": "Buat"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "apo", "output": "Apa"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "aie", "output": "Air"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Esok saya balik kampung", "output": "Esok kome balek kampung"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Semalam saya datang lambat", "output": "Semalam kome datang lambak"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "lambak", "output": "Lambat"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kampung", "output": "Kampung / kampung halaman"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ikan itu segar", "output": "Ikan tu sego"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Laut itu indah", "output": "Lauk tu indah"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pantai cantik", "output": "Pantai cantik"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "lauk", "output": "Laut"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "sego", "output": "Segar"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak beli ikan", "output": "Kome nok beli ikan"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu pergi pasar?", "output": "Demo gi paso?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ya, saya pergi pasar", "output": "Yo, kome gi paso"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "paso", "output": "Pasar"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "yo", "output": "Ya / betul"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Tidak, saya tidak pergi", "output": "Tak, kome tak gi"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia pandai masak", "output": "Dia pandai masok"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya tidak pandai masak", "output": "Kome tak pandai masok"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "masok", "output": "Masak"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "pandai", "output": "Pandai / pandai melakukan sesuatu"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Rumah saya dekat pantai", "output": "Rumah kome dokat pantai"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Jauh sangat", "output": "Jauh sangat"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "dokat", "output": "Dekat / berhampiran"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya letih hari ini", "output": "Kome letih ari ni"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu rehat dulu", "output": "Demo rehat dulu"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "letih", "output": "Letih / penat"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ayam itu gemuk", "output": "Ayam tu gemok"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kucing itu kurus", "output": "Kucing tu kuru"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "gemok", "output": "Gemuk"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kuru", "output": "Kurus"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya gembira hari ini", "output": "Kome gumbiro ari ni"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia sedih sangat", "output": "Dia sedeh sangat"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "gumbiro", "output": "Gembira / senang hati"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "sedeh", "output": "Sedih"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya sakit kepala", "output": "Kome sakik kapalo"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pergi jumpa doktor", "output": "Gi jumpe dokto"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "sakik", "output": "Sakit"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kapalo", "output": "Kepala"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "dokto", "output": "Doktor"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu mahu makan apa?", "output": "Demo nok makang apo?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya mahu makan nasi goreng", "output": "Kome nok makang nasi goring"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "goring", "output": "Goreng"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Tolong jangan bising", "output": "Tolong jangan bising"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak tidur awal", "output": "Kome nok tiduq awal"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Bangun pagi esok", "output": "Bangun pagi esok"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia lambat datang", "output": "Dia lambak datang"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya sudah siap", "output": "Kome dah siap"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu belum siap lagi", "output": "Demo belom siap lagi"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Cepat sikit", "output": "Cepat siket"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Perlahan sikit", "output": "Pahan siket"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "pahan", "output": "Perlahan / perlahan"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya suka makan ikan bakar", "output": "Kome suko makang ikan bakaq"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ikan bakar sedap", "output": "Ikan bakaq sedap"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "bakaq", "output": "Bakar"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "sedap", "output": "Sedap / enak"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu sudah makan tengah hari?", "output": "Demo dah makang tanghaghie?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Sudah makan tengah hari", "output": "Dah makang tanghaghie"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "tanghaghie", "output": "Tengah hari"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak minum petang", "output": "Kome nok minum petang"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Malam ini gelap", "output": "Malam ni gelap"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Bintang banyak malam ini", "output": "Bintang banyak malam ni"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Bulan terang malam ini", "output": "Bulan terang malam ni"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "terang", "output": "Terang / bersinar"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya sayang kamu", "output": "Kome sayang demo"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu sayang saya tak?", "output": "Demo sayang kome tak?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Keluarga saya besar", "output": "Kaluargo kome beso"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kaluargo", "output": "Keluarga"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ayah saya pergi kerja", "output": "Ayah kome gi kejo"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ibu saya masak di dapur", "output": "Mak kome masok di dapo"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "dapo", "output": "Dapur"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Adik saya pergi sekolah", "output": "Adik kome gi sekola"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "sekola", "output": "Sekolah"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya belajar di universiti", "output": "Kome belajar di universiti"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu belajar apa?", "output": "Demo belajar apo?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya belajar komputer", "output": "Kome belajar komputer"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia sangat rajin", "output": "Dia rajin sangat"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "rajin", "output": "Rajin / tekun"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Jangan malas", "output": "Jangan malaih"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "malaih", "output": "Malas"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak pergi kedai", "output": "Kome nok gi kedai"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kedai itu jauh", "output": "Kedai tu jauh"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Jalan itu sempit", "output": "Jalan tu sempit"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kereta itu laju", "output": "Kereta tu laju"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "laju", "output": "Laju / cepat"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya naik basikal", "output": "Kome naik basikal"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia naik motosikal", "output": "Dia naik motosikal"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Tolong tunggu saya", "output": "Tolong tunggu kome"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "tunggu", "output": "Tunggu"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya sudah sampai", "output": "Kome dah sampai"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia belum sampai lagi", "output": "Dia belom sampai lagi"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu di mana?", "output": "Demo di mano?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya di rumah", "output": "Kome di rumah"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia di pasar", "output": "Dia di paso"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kita pergi bersama", "output": "Kite gi besamo"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "besamo", "output": "Bersama / bersama-sama"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya tidak mahu pergi", "output": "Kome tak nok gi"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia mahu ikut", "output": "Dia nok ikuk"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "ikuk", "output": "Ikut"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Hari ini hari Jumaat", "output": "Ari ni ari Jumaat"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Esok hari Sabtu", "output": "Esok ari Sabtu"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Semalam hari Khamis", "output": "Semalam ari Khamis"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak tidur tengah hari", "output": "Kome nok tiduq tanghaghie"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Bangun tidur petang", "output": "Bangun tiduq petang"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Cuaca cantik hari ini", "output": "Cuaco cantik ari ni"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "cuaco", "output": "Cuaca"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Angin kuat hari ini", "output": "Angin kuat ari ni"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Awan gelap di langit", "output": "Awan gelap di langik"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "langik", "output": "Langit"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak pergi memancing", "output": "Kome nok gi mancing"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pancing ikan di sungai", "output": "Mancing ikan di sungoi"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "mancing", "output": "Memancing / menangkap ikan"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "sungoi", "output": "Sungai"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ikan besar di laut", "output": "Ikan beso di lauk"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya suka pantai", "output": "Kome suko pantai"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pasir putih bersih", "output": "Pasi puteh besih"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "pasi", "output": "Pasir"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "puteh", "output": "Putih"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "besih", "output": "Bersih"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Terengganu negeri saya", "output": "Ganu nighi kome"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya bangga jadi orang Terengganu", "output": "Kome banggo jadi oghe Ganu"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "banggo", "output": "Bangga"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Loghat Terengganu unik", "output": "Loghat Ganu unik"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya belajar loghat Terengganu", "output": "Kome belajar loghat Ganu"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu faham loghat Terengganu?", "output": "Demo paham loghat Ganu?"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Sikit-sikit saya faham", "output": "Siket-siket kome paham"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia fasih loghat Terengganu", "output": "Dia fasih loghat Ganu"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "fasih", "output": "Fasih / lancar bercakap"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak belajar lagi", "output": "Kome nok belajar lagi"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu ajar saya", "output": "Demo ajo kome"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "ajo", "output": "Ajar"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Terima kasih banyak", "output": "Teimo kasih banyak"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Sama-sama, jangan lupa", "output": "Samo-samo, jangan lupo"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "lupo", "output": "Lupa"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya tidak akan lupa", "output": "Kome takkan lupo"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kita jumpa lagi esok", "output": "Kite jumpe lagi esok"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Selamat tinggal", "output": "Selamat tinggal"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Selamat jalan", "output": "Selamat jalan"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Jumpa lagi", "output": "Jumpe lagi"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya gembira dapat kenal kamu", "output": "Kome gumbiro dapat kenal demo"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kamu kawan saya", "output": "Demo kawan kome"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "kawan", "output": "Kawan / teman"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kita kawan baik", "output": "Kite kawan baik"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Jangan lupa kawan", "output": "Jangan lupo kawan"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya rindu kamu", "output": "Kome rindu demo"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "rindu", "output": "Rindu / rindu rindu"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Dia rindu kampung halaman", "output": "Dia rindu kampung halamon"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "halamon", "output": "Halaman"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pulang ke kampung", "output": "Pulang ke kampung"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya rindu masakan ibu", "output": "Kome rindu masok mak"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Masakan Terengganu sedap", "output": "Masak Ganu sedap"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Nasi dagang paling sedap", "output": "Nasi dagang paling sedap"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "dagang", "output": "Dagang (nasi dagang - hidangan tradisional Terengganu)"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Keropok lekor makanan kegemaran", "output": "Keropok leko makonon kegemaran"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "leko", "output": "Lekor (keropok lekor - makanan tradisional Terengganu)"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "makonon", "output": "Makanan"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya suka keropok lekor", "output": "Kome suko keropok leko"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Beli keropok di Kedai Tepi Jalan", "output": "Beli keropok di Kedai Tepi Jalan"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Keropok itu rangup", "output": "Keropok tu rangup"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "rangup", "output": "Rangup / garing"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Cicah dengan kuah kacang", "output": "Cicah dengan kuah kacang"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "cicah", "output": "Cicah / celup"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak pergi Kuala Terengganu", "output": "Kome nok gi Kuala Ganu"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pasar Payah di Kuala Terengganu", "output": "Paso Payah di Kuala Ganu"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "payah", "output": "Payah (Pasar Payah - pasar terkenal di Terengganu)"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pantai Batu Buruk cantik", "output": "Pantai Batu Buruk cantik"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pulau Perhentian indah", "output": "Pulau Perhentian indah"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Air laut jernih dan biru", "output": "Aie lauk jenih dan biru"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "biru", "output": "Biru"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak snorkeling", "output": "Kome nok snorkeling"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Terumbu karang cantik", "output": "Terumbu karang cantik"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Ikan warna-warni di laut", "output": "Ikan warno-warni di lauk"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "warno", "output": "Warna"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya takut ikan besar", "output": "Kome takuk ikan beso"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "takuk", "output": "Takut"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Jangan takut, ikan kecil saja", "output": "Jangan takuk, ikan kecik je"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Penyu ada di sini", "output": "Penyu ado di sini"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "ado", "output": "Ada"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Pulau Redang juga cantik", "output": "Pulau Redang jugo cantik"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "jugo", "output": "Juga"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya suka semua pulau", "output": "Kome suko semua pulau"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Terengganu negeri paling cantik", "output": "Ganu nighi paling cantik"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya nak duduk sini selamanya", "output": "Kome nok duduk sini slamenye"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "slamenye", "output": "Selamanya"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Hidup orang Terengganu", "output": "Hidup oghe Ganu"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Saya sayang Terengganu", "output": "Kome sayang Ganu"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Terengganu maju dan pesat", "output": "Ganu maju dan pesat"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "pesat", "output": "Pesat / berkembang cepat"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Maju ke hadapan", "output": "Maju ke hadapan"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Jangan tinggal di belakang", "output": "Jangan tinggal di belakang"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kita semua bersatu", "output": "Kite semua besatu"},
{"instruction": "Apakah maksud perkataan ini dalam loghat Terengganu", "input": "besatu", "output": "Bersatu"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Kita kuat bersama", "output": "Kite kuat besamo"},
{"instruction": "Tukarkan ayat ini ke loghat Terengganu", "input": "Bersama kita boleh", "output": "Besamo kite boleh"},

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
