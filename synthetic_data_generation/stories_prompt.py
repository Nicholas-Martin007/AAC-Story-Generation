STORY_PROMPT = r"""
Kamu adalah AI yang membuat *Kisah Sosial* (cerita pendek satu paragraf) yang ditujukan untuk anak-anak.

Petunjuk:
1. Tulis satu paragraf berupa kisah sosial yang **positif**, **mendidik**, dan **aman untuk anak-anak**.
2. Paragraf harus terdiri dari **5 sampai 10 kalimat sederhana**.
3. Gunakan **kata-kata yang mudah dimengerti oleh anak-anak**. Hindari kata sulit, bahasa gaul, atau ajakan langsung seperti "ayo", "yuk", atau "kamu harus".
4. Gunakan gaya bahasa yang **tenang, ramah, dan menyenangkan**.
5. Semua nama orang diganti menjadi:
   - <|PER|>, <|PER_1|>, dst sesuai urutan kemunculan, atau gunakan “dia”.
6. Jika tersedia, gunakan **beberapa kata dari daftar `context`** sebagai bagian dari cerita. Tidak harus semua digunakan.
7. Jika `context` kosong, tetap buat kisah sesuai dengan aturan.
8. Jangan gunakan dialog atau tanda kutip.
9. Hanya tampilkan satu baris dengan format:
   Story: ...
10. Gunakan kreativitasmu selama tetap sesuai aturan dan aman untuk anak-anak.

Format Output:
Story: ...

Contoh:
"""

LIST_STORIES_PROMPT = [
    {
        "content": "<|PER|> bangun pagi dan langsung merapikan tempat tidurnya. Setelah itu, dia mencuci tangan sebelum sarapan bersama keluarganya. Ia merasa senang karena bisa membantu di rumah. Keluarganya tersenyum melihat sikapnya. Kegiatan pagi hari jadi lebih menyenangkan saat semuanya rapi dan bersih. <|PER|> belajar bahwa memulai hari dengan baik membuat semuanya terasa lebih mudah."
    },
    {
        "content": "Setiap hari, <|PER|> berjalan kaki ke sekolah. Di jalan, dia selalu menyapa teman-temannya dengan senyum. Ia juga melambaikan tangan dan memberi salam. Teman-temannya senang mendapat sapaan darinya. Guru di sekolah bilang, <|PER|> anak yang ramah. Saling menyapa membuat hari terasa lebih ceria dan hangat."
    },
    {
        "content": "Saat istirahat, <|PER|> melihat temannya kesulitan membuka kotak makan. Tanpa diminta, dia langsung membantu. Temannya tersenyum dan berterima kasih. Mereka lalu makan bersama sambil bercerita. <|PER|> merasa senang bisa membantu. Ia belajar bahwa menolong orang lain membuat hati jadi hangat."
    },
    {
        "content": "<|PER_1|>, <|PER|>, dan <|PER_2|> bermain bersama di taman. Mereka bergiliran naik ayunan dan tidak saling berebut. Mereka tertawa dan saling menyemangati. Semua anak merasa senang karena bisa bermain bersama dengan adil. Mereka belajar bahwa sabar dan saling menghargai membuat bermain jadi lebih menyenangkan."
    },
    {
        "content": "<|PER|> sedang sedih karena mainannya rusak. Ibunya memeluknya dan berkata bahwa tidak apa-apa merasa sedih. Setelah itu, mereka mencoba memperbaikinya bersama-sama. Meskipun tidak berhasil, <|PER|> merasa lebih baik. Ia belajar bahwa perasaan sedih itu wajar dan bisa dihadapi bersama orang yang sayang padanya."
    },
    {
        "content": "<|PER|> melihat sampah di lantai kelas. Tanpa disuruh, dia memungut dan membuangnya ke tempat sampah. Teman-temannya melihat dan ikut melakukan hal yang sama. Kelas menjadi bersih dan nyaman. Guru memuji sikap mereka. <|PER|> merasa bangga bisa memberi contoh yang baik."
    },
    {
        "content": "Ketika bermain bola, <|PER|> tanpa sengaja membuat temannya jatuh. Dia langsung menghampiri dan meminta maaf. Temannya menerima permintaan maaf itu dan tersenyum. <|PER|> membantu temannya berdiri, lalu mereka bermain lagi bersama. Ia belajar bahwa bertanggung jawab atas kesalahan adalah hal yang penting."
    },
    {
        "content": "<|PER|> membawa bekal lebih dan membaginya dengan temannya yang lupa membawa makan siang. Mereka makan bersama sambil tersenyum. Temannya merasa senang dan berterima kasih. <|PER|> merasa bahagia bisa berbagi. Ia belajar bahwa berbagi membuat orang lain dan dirinya sama-sama senang."
    },
    {
        "content": "Saat guru bertanya, <|PER|> mengangkat tangan dengan sopan. Ia menjawab dengan suara pelan tapi jelas. Guru tersenyum dan memuji usahanya. Teman-temannya ikut memperhatikan. <|PER|> merasa percaya diri karena telah berani mencoba. Ia belajar bahwa mengikuti aturan membuat suasana kelas jadi menyenangkan."
    },
    {
        "content": "<|PER_1|> melihat <|PER_2|> sendirian di sudut kelas. Dia menghampiri dan mengajak bermain bersama. <|PER_2|> tersenyum dan bergabung dalam permainan. Sejak saat itu, mereka sering bermain bersama. Mereka menjadi teman baik. <|PER_1|> belajar bahwa mengajak teman adalah hal baik yang bisa membuat orang lain merasa senang."
    }
]
