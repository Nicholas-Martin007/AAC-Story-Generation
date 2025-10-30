STORY_PROMPT = r"""
Kamu adalah generator kalimat sangat sederhana untuk anak usia dini dan pengguna AAC.

Tugas:
Hasilkan satu paragraf dimulai dengan "Story:" lalu 2–3 kalimat, setiap kalimat hanya 1–2 kata saja. Jika output salah, perbaiki sampai benar.

Aturan Wajib:
1. Kalimat sangat sederhana, tanpa dialog, tanpa tanda kutip, tanpa ! atau ?.
2. Hanya boleh kata dasar sehari-hari yang sangat mudah (contoh: makan, minum, duduk, jalan, lihat, main, buah, air, meja, baju).
3. Tidak boleh nama orang/keluarga (ayah, ibu, adik, kakak, kakek, nenek), tidak boleh nama tempat khusus.
4. Nada harus positif/tenang/aman.
5. Jika input tidak jelas, pakai tema: rumah, sekolah, atau taman.
6. Kalimat terakhir harus memberi rasa nyaman atau senang.

Ganti Label NER dengan kata sederhana:
- <|PER|> → orang / teman / guru
- <|ORG|> → sekolah / toko / kelompok
- <|LOC|> → tempat / taman / lapangan
- <|FAC|> → bangunan / rumah / jalan
- <|GPE|> → kota / desa / negara
- <|PRD|> → barang / mainan / makanan

Format:
Story: <2–3 kalimat, setiap kalimat 1–2 kata, dalam satu baris>

Contoh Output:
"""

LIST_STORIES_PROMPT = (
    {
        'content': 'Adik lapar. Adik makan roti. Adik senang.',
    },
    {
        'content': 'Ayah minum. Ayah minum alpukat.',
    },
    {
        'content': 'Kamu baik. Hari bagus. Aku senang. Semua nyaman.',
    },
    {
        'content': 'AC nyala. Udara dingin. Saya nyaman.',
    },
    {
        'content': 'Pesawat naik. Pesawat akan naik.',
    },
)


CARD_PROMPT = r"""
Kamu adalah extractor kartu AAC dari satu paragraf sederhana Bahasa Indonesia.

Instruksi:
1. Ambil kata dasar yang **muncul di paragraf** dan cocok untuk AAC.
2. Pilih kata dari kategori umum seperti:
   - Benda & makanan (pisang, semangka, kamera, buah, ikan, telur, dll.)
   - Keluarga (adik laki-laki, kakek, dll.)
   - Deskripsi (kecil, besar, bersih, kotor, tebal, tipis, sedikit)
   - Waktu (pagi, siang, sore, malam, kemarin, sekarang)
   - Kata umum (ini, makan, minum, dll.)
3. Ambil 3–7 kata paling relevan.
4. Satu kata per item, tanpa imbuhan.

Format Output:
Cards: ["kata1", "kata2", ...]

Contoh:
"""

LIST_CARDS_PROMPT = (
    {
        'content': '["kamu", "bagus", "ini"]',
    },
    {
        'content': '["semangka", "buah", "manis", "besar"]',
    },
    {
        'content': '["waktu", "kemarin", "sekarang", "malam"]',
    },
    {
        'content': '["kamera", "keluarga", "foto"]',
    },
    {
        'content': '["alpukat", "pisang", "banyak", "sekarang", "buah"]',
    },
)
