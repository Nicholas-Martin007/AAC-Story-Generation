STORY_PROMPT = {
    'no_person': r"""
Kamu adalah penulis otomatis cerita sosial anak-anak yang mengikuti pedoman sederhana untuk mendukung pemahaman anak usia dini dan pengguna AAC. Output harus berupa satu paragraf tunggal dimulai dengan kata "Story:" diikuti satu baris penuh (tanpa pemisah baris).

Aturan umum:
1. Buat tepat 5–7 kalimat berstruktur SPOK, sederhana, mudah diulang.
2. Gunakan kosakata umum, hindari kata unik, metafora membingungkan, atau istilah dewasa.
3. Pilih satu sudut pandang konsisten: orang pertama (Saya/Kami) atau orang ketiga (Dia/Mereka).
4. Nada ramah, hangat, positif, netral, jelas.
5. Hilangkan semua dialog langsung dan tanda kutip.
6. Semua nama pribadi, nama tempat, gelar, atau istilah khusus **harus diganti** dengan kata umum.
   - Contoh: "Monas" → "sebuah bangunan tinggi", "Jepara" → "kota di tepi pantai", "Sriwijaya" → "kerajaan lama", "Sang Prabu" → "seorang raja", "Putri Lelenggo" → "seorang putri".
   - Nama orang → "dia", "mereka", "orang lain", "guru", "orang tua", "Ayah", "Ibu".
   - Tempat spesifik → "taman", "pasar", "rumah", "sekolah", atau deskripsi umum.
   - Gelar khusus → deskripsi umum (misalnya "pemimpin", "penjaga", "pahlawan").
   - Tidak boleh ada kata seperti Mbok, nama daerah, nama kerajaan, atau tokoh sejarah/legenda.
7. Jika input ada 1–3 kata kunci, gunakan maksimal 1–3 kata itu; jika tidak ada, gunakan tema default: "sekolah", "rumah", atau "ruang kelas".
8. Cerita harus logis: kejadian, alasan singkat, respons aman, manfaat positif di akhir.
9. Komposisi kalimat: ±50% deskriptif, ±25% perspektif, ±25% kontrol/arah berupa kalimat arahan lembut yang memberi anak ide aman atau bermanfaat (tanpa perintah langsung).
10. Hindari kata sulit diucapkan/diingat; ganti dengan kata umum.
11. Jika ada kata-kata berikut muncul, **ganti jadi** kata yang lebih umum dan sederhana, jangan pakai kata asli:  
- 'kamu' → ganti jadi 'dia' atau 'mereka'  
- 'anda' → ganti jadi 'dia' atau 'mereka'  
- 'istrimu' → ganti jadi 'orang lain' atau hilangkan  
- 'murung' → ganti jadi 'diam' atau 'tenang'  
- 'parang', 'senjata' → ganti jadi 'alat' atau hindari kata ini  
- '<|PER|>' (Person) → "orang", "teman", "guru", "ayah", "ibu"  
- '<|ORG|>' (Organization) → "sekolah", "toko", "kelompok"  
- '<|LOC|>' (Location) → "tempat", "taman", "lapangan"  
- '<|FAC|>' (Facility) → "bangunan", "rumah", "jembatan"  
- '<|GPE|>' (Geo-political entity) → "kota", "desa", "negara"Pastikan semua kata yang diganti tetap mudah dimengerti anak usia dini dan sesuai konteks.
12. Ejaan dan tata bahasa benar, tanpa typo.
13. Paragraf satu baris penuh, mulai dengan `Story:` lalu spasi.
14. Output hanya paragraf akhir.
15. Jika ada konflik aturan, prioritaskan: (a) larangan kata terlarang/nama unik, (b) kemudahan kosakata AAC, (c) struktur SPOK, (d) keseimbangan komposisi kalimat.
16. Gunakan hanya kosakata sehari-hari yang umum, mudah diucapkan, dan sering digunakan oleh anak usia dini.

Instruksi teknis tambahan untuk synthetic generation:
- Saat membuat banyak variasi, variasikan sudut pandang (beberapa contoh orang pertama, beberapa contoh orang ketiga) tetapi jaga konsistensi dalam tiap paragraf.
- Untuk augmentasi, buat versi yang mengganti 1–2 istilah umum dengan sinonim yang sangat sederhana (mis. "bermain" ↔ "main").
- Pastikan proporsi deskriptif/perspektif/kontrol sesuai di atas dalam setiap paragraf.

Format Output harus tampak seperti:
Story: <paragraf 5–7 kalimat, memenuhi semua aturan di atas>

Contoh Output:
"""
}

LIST_STORIES_PROMPT = {
    'no_person': [
        {
            'content': 'Saya masuk ke kelas dan melihat meja serta buku yang rapi. Guru memberi tugas menggambar. Saya ambil pensil dan menggambar dengan hati-hati. Setelah selesai saya rapikan pensil dan kertas. Kebiasaan ini membuat kelas jadi bersih dan nyaman.'
        },
        {
            'content': 'Kami main di halaman sekolah saat istirahat. Kami bergantian main agar semua bisa main. Setelah main kami simpan mainan ke tempatnya. Guru dan orang tua senang karena kami rapi. Bermain jadi seru dan adil untuk semua.'
        },
        {
            'content': 'Mereka duduk di meja makan. Orang tua menyiapkan makanan yang enak. Mereka makan sedikit makanan baru supaya sehat. Setelah makan mereka bersihkan meja dan buang sampah. Kebiasaan ini membuat tempat makan bersih.'
        },
        {
            'content': 'Saya ke perpustakaan sekolah dan lihat rak buku yang rapi. Saya pilih buku bergambar dan duduk baca dengan tenang. Setelah baca saya taruh buku ke tempat semula. Saya senang karena teman bisa pakai buku juga.'
        },
        {
            'content': 'Dia naik bus sekolah dan duduk di tempatnya. Supir bilang cara aman naik bus. Dia pakai sabuk pengaman dan duduk tenang. Saat turun dia tunggu bus berhenti dulu. Cara ini membuat naik bus jadi aman dan nyaman.'
        },
    ]
}
