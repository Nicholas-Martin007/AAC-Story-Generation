CARD_PROMPT = r"""
Kamu adalah extractor *AAC cards* dalam bentuk array dari satu paragraf Bahasa Indonesia yang ditulis memakai kata-kata dasar.

Instruksi:
1. Identifikasi kosakata umum dari paragraf yang cocok untuk *Augmentative and Alternative Communication (AAC)*, meliputi:
   - Aktivitas sehari-hari (contoh: makan, tidur, jalan, main, baca)
   - Emosi (contoh: senang, sedih, marah, takut, tenang)
   - Benda umum (contoh: meja, kursi, buku, tas, bola)
   - Tempat umum (contoh: rumah, sekolah, taman, kelas, halaman)
   - Orang → gunakan kata ganti sesuai cerita (contoh: dia, mereka, saya, kami)
2. Gunakan hanya kata dasar (1 kata per item, tanpa imbuhan).
3. Pilih jumlah kata yang relevan secara alami dengan cerita, pilih 5 atau 7 atau 10 kata. 
   - Jika hanya sedikit kosakata penting, ambil sedikit saja (misalnya 2–4 kata).
   - Jangan selalu mengambil jumlah maksimal.
4. Jangan membuat kata baru atau menggabungkan kata.
5. Output hanya dalam format array Python, tanpa penjelasan lain.

Format Output:
Cards: "["kata1", "kata2", ...]"

Contoh:
"""
LIST_CARDS_PROMPT = (
    {'content': '["taman", "main", "senang"]'},
    {'content': '["dia", "makan", "meja", "kursi"]'},
    {
        'content': '["mereka", "bola", "lari", "air", "teman", "senang", "taman"]'
    },
    {
        'content': '["kami", "sekolah", "kelas", "baca", "buku", "tas"]'
    },
    {'content': '["saya", "tenang", "tas"]'},
)
