import torch

DATA_PATH = "data.json"
N_SHOTS = 10

# MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# NER_MODEL_PATH = "cahya/bert-base-indonesian-NER"
NER_MODEL_PATH = "cahya/NusaBert-ner-v1.3"
# MODEL_PATH = "/home/dev/Downloads/Llama-3.2-3B-Instruct/"

MODEL_PATH = "/home/dev/Downloads/Meta-Llama-3.1-8B-Instruct/"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    "generation": {
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9,
    },
}



PROMPT = r"""
Kamu adalah extractor data untuk pelatihan NLP dengan format *AAC Cards* dari satu paragraf *Story* Bahasa Indonesia.

Tugasmu:
1. Ambil kata-kata penting dari paragraf Story, seperti subjek, predikat, objek, atau keterangan.
2. Hasilkan output dalam bentuk array Python seperti: ["aku", "rumah", "berjalan"]
3. Kata boleh duplikat jika muncul lebih dari sekali dan relevan secara makna.
4. JANGAN ambil kata hubung (dan, atau, tetapi, lalu, karena, meskipun, dll) atau kata bantu (telah, akan, sedang, dll).
5. Gunakan hanya kata baku Bahasa Indonesia, bukan kata tidak baku, dialek, atau bahasa daerah.
6. Jika Story mengandung token nama orang seperti <|PER|>, <|PER_1|>, dst, maka token tersebut WAJIB dimasukkan ke dalam hasil.
7. JANGAN ambil nama tempat, organisasi, atau nama khusus lainnya (contoh: "Teater Gandrik", "Indonesia Kita", "Bagong", dll).
8. Jangan ubah isi Story. Fokus hanya mengambil kata dari Story tersebut.
9. Jika tidak bisa menghasilkan sesuai aturan, lewati saja. Jangan tampilkan pesan atau penjelasan apa pun.
10. Jumlah AAC Cards minimal 1 dan maksimal 20 kata. Jika lebih dari 20, pilih yang paling relevan atau paling sering muncul.

Format Output:
AAC Cards: ["kata1", "kata2", "kata3", ...]

Contoh Output:
"""