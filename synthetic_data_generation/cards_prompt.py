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
    {'content': '["kamu", "bagus", "ini"]'},
    {'content': '["semangka", "buah", "manis", "besar"]'},
    {'content': '["waktu", "kemarin", "sekarang", "malam"]'},
    {'content': '["kamera", "keluarga", "foto"]'},
    {
        'content': '["alpukat", "pisang", "banyak", "sekarang", "buah"]'
    },
)
