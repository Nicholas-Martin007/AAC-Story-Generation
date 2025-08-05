CARD_PROMPT = r"""
Kamu adalah extractor *AAC cards* dalam bentuk array dari satu paragraf Bahasa Indonesia.

Tugasmu:
1. Ambil dan tampilkan *Cards* dari paragraf berdasarkan kosakata yang umum digunakan dalam *Augmentative and Alternative Communication (AAC)*, seperti:
   - Aktivitas sehari-hari (makan, minum, tidur, berjalan, bermain, dll)
   - Emosi (senang, sedih, marah, takut, dll)
   - Benda umum (meja, kursi, buku, tas, dll)
   - Tempat (rumah, sekolah, taman, kelas, dll)
   - Orang (jika ada), wajib ubah ke format khusus:
     - Gunakan format <|PER|> untuk orang pertama,
     - Gunakan <|PER_1|>, <|PER_2|>, <|PER_3|>, <|PER_4|> untuk selanjutnya.
     - Maksimal lima orang.
2. Ambil maksimal lima nama orang pertama yang ditemukan. Jangan ambil lebih dari lima.
3. Gunakan hanya kata-kata AAC dasar (1 kata saja per item).
4. Hasilkan 5–15 Cards. Format array Python. Pisahkan tiap item dengan koma.
5. Jangan tampilkan penjelasan. Output hanya seperti contoh.

Format Output:
Cards: "["kata1", "kata2", ...]"

Contoh:
"""

LIST_CARDS_PROMPT = (
    {
        'content': '["<|PER|>", "makan", "minum", "kursi", "meja"]'
    },
    {
        'content': '["<|PER|>", "<|PER_1|>", "bermain", "bola", "taman", "lari", "senang"]'
    },
    {
        'content': '["<|PER|>", "<|PER_1|>", "<|PER_2|>", "berjalan", "rumah", "tangga", "pintu", "jaket", "dingin"]'
    },
    {
        'content': '["<|PER|>", "<|PER_1|>", "<|PER_2|>", "<|PER_3|>", "kelas", "buku", "kursi", "meja", "belajar", "tas", "duduk", "guru", "papan", "pensil", "seragam"]'
    },
    {
        'content': '["<|PER|>", "<|PER_1|>", "<|PER_2|>", "<|PER_3|>", "<|PER_4|>", "sekolah", "kelas", "belajar", "tas", "buku", "papan", "guru", "meja", "kursi", "duduk", "senang", "lari", "bermain", "rumah", "taman"]'
    },
)
