CARD_PROMPT = r"""
Kamu adalah extractor *AAC cards* dalam bentuk array dari satu paragraf Bahasa Indonesia.

Tugasmu:
1. Ambil dan tampilkan *Cards* dari paragraf berdasarkan kosakata yang umum digunakan dalam *Augmentative and Alternative Communication (AAC)*, seperti:
   - Aktivitas sehari-hari (makan, minum, tidur, berjalan, dll)
   - Emosi (senang, sedih, marah, takut, dll)
   - Benda umum (meja, kursi, baju, buku, dll)
   - Tempat (rumah, sekolah, taman, dll)
   - Orang (jika ada), ubah ke format <|PER|>, <|PER_1|>, dst
2. Jika ada nama orang, ubah ke format  <|PER|>, <|PER_1|>, dst. Jika tidak ada, tidak masalah.
3. Gunakan hanya kata-kata yang termasuk dalam kosakata AAC dasar (hindari kata abstrak atau teknis seperti “politik”, “program”, “kenangan”).
4. Hasilkan 5–15 *Cards* berupa 1 kata saja, pisahkan dengan koma.
5. Tampilkan hanya *Cards*, tanpa penjelasan atau format lain.

Format Output:
Cards: "["kata1", "kata2", ...]"

Contoh:
"""

LIST_CARDS_PROMPT = [
    {"content": '["<|PER|>", "<|PER_1|>", "ruang", "lihat", "jelas"]'},
    {"content": '["<|PER|>", "jendela", "sejarah"]'},
    {"content": '["bayangan", "malam"]'},
    {"content": '["<|PER|>", "<|PER_1|>", "<|PER_2|>", "rumah", "sunyi"]'},
    {"content": '["<|PER|>", "malam", "istri"]'},
    {"content": '["<|PER|>", "rumah", "ikan", "sore"]'},
    {"content": '["ruangan", "wewangian", "aroma", "keyakinan", "orang"]'},
    {"content": '["<|PER|>", "<|PER_1|>", "<|PER_2|>", "taman", "senja", "kabar"]'},
    {"content": '["<|PER|>", "kota", "kematian", "percakapan"]'},
    {"content": '["<|PER|>", "aktor", "Teater"]'},
]
