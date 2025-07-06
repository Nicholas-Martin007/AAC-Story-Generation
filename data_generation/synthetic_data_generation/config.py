N_SHOTS = 5

MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"

PROMPT = """
Kamu adalah generator data yang membuat pasangan input-output terstruktur untuk pelatihan NLP.

Tugasmu:
1. Ambil kata kunci penting dari sebuah paragraf (bisa berupa subjek, predikat, objek, atau keterangan).
2. Pilih 2–10 kata kunci secara acak dari teks.
3. Semua kata kunci ditulis dalam huruf kecil dan dalam bentuk dasar (tidak perlu dikonjugasi atau dibentuk ulang).
4. Buat paragraf singkat (2–5 kalimat) yang koheren, alami, dan masuk akal berdasarkan kata kunci tersebut.
5. Kamu boleh menambahkan informasi logis agar konteks lengkap. Gunakan variasi struktur kalimat dan Bahasa Indonesia yang baik dan benar.
6. Hindari pengulangan dan gunakan sinonim seperlunya untuk membuat paragraf terasa alami.

Format Output:
Input: ["kata1", "kata2", ..., "kataN"]
Output: Paragraf berdasarkan kata kunci.

Contoh:
"""


MODEL_CONFIG = {
    "generation": {
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9,
    },
}