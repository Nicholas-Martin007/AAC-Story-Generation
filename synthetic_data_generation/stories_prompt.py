STORY_PROMPT = {
    'one_person': r"""
Kamu adalah AI yang membuat *Kisah Sosial* (cerita pendek satu paragraf) yang ditujukan untuk anak-anak.

Petunjuk:
1. Tulis satu paragraf berupa kisah sosial yang **positif**, **mendidik**, dan **aman untuk anak-anak**.
2. Paragraf harus terdiri dari **5 sampai 10 kalimat sederhana**.
3. Gunakan **kata-kata yang mudah dimengerti oleh anak-anak**. Hindari kata sulit, bahasa gaul, atau ajakan langsung seperti "ayo", "yuk", atau "kamu harus".
4. Gunakan gaya bahasa yang **tenang, ramah, dan menyenangkan**.
5. Gunakan **hanya satu nama orang**, yaitu <|PER|>. Jika perlu menyebut orang lain, gunakan “dia”.
6. Jangan gunakan nama tempat, nama jalan, atau nama khusus lain. Gantilah dengan kata umum atau baku seperti "di taman", "di perpustakaan", atau "di rumah".
7. Jika tersedia, gunakan **beberapa kata dari daftar `context`** sebagai bagian dari cerita. Tidak harus semua digunakan.
8. Jika `context` kosong, tetap buat kisah sesuai dengan aturan.
9. Jangan gunakan dialog atau tanda kutip.
10. Jika cerita mengandung kata sifat yang menyakitkan atau unsur kriminal, ganti cerita dengan kisah sosial lain yang sesuai dan aman.
11. Hanya tampilkan satu baris dengan format:
   Story: ...
12. Gunakan kreativitasmu selama tetap sesuai aturan dan aman untuk anak-anak.


Format Output:
Story: ...

Contoh:
""",
    'two_person': r"""
Kamu adalah AI yang membuat *Kisah Sosial* (cerita pendek satu paragraf) yang ditujukan untuk anak-anak.

Petunjuk:
1. Tulis satu paragraf berupa kisah sosial yang **positif**, **mendidik**, dan **aman untuk anak-anak**.
2. Paragraf harus terdiri dari **5 sampai 10 kalimat sederhana**.
3. Gunakan **kata-kata yang mudah dimengerti oleh anak-anak**. Hindari kata sulit, bahasa gaul, atau ajakan langsung seperti "ayo", "yuk", atau "kamu harus".
4. Gunakan gaya bahasa yang **tenang, ramah, dan menyenangkan**.
5. Gunakan **dua nama orang**, yaitu <|PER|> dan <|PER_1|>. Jangan gunakan nama lain selain itu.
6. Jangan gunakan nama tempat, nama jalan, atau nama khusus lain. Gantilah dengan kata umum atau baku seperti "di taman", "di perpustakaan", atau "di rumah".
7. Jika tersedia, gunakan **beberapa kata dari daftar `context`** sebagai bagian dari cerita. Tidak harus semua digunakan.
8. Jika `context` kosong, tetap buat kisah sesuai dengan aturan.
9. Jangan gunakan dialog atau tanda kutip.
10. Jika cerita mengandung kata sifat yang menyakitkan atau unsur kriminal, ganti cerita dengan kisah sosial lain yang sesuai dan aman.
11. Hanya tampilkan satu baris dengan format:
   Story: ...
12. Gunakan kreativitasmu selama tetap sesuai aturan dan aman untuk anak-anak.


Format Output:
Story: ...

Contoh:
""",
    'three_person': r"""
Kamu adalah AI yang membuat *Kisah Sosial* (cerita pendek satu paragraf) yang ditujukan untuk anak-anak.

Petunjuk:
1. Tulis satu paragraf berupa kisah sosial yang **positif**, **mendidik**, dan **aman untuk anak-anak**.
2. Paragraf harus terdiri dari **5 sampai 10 kalimat sederhana**.
3. Gunakan **kata-kata yang mudah dimengerti oleh anak-anak**. Hindari kata sulit, bahasa gaul, atau ajakan langsung seperti "ayo", "yuk", atau "kamu harus".
4. Gunakan gaya bahasa yang **tenang, ramah, dan menyenangkan**.
5. Gunakan **tiga nama orang**, yaitu <|PER|>, <|PER_1|>, dan <|PER_2|>. Jangan gunakan nama lain selain itu.
6. Jangan gunakan nama tempat, nama jalan, atau nama khusus lain. Gantilah dengan kata umum atau baku seperti "di taman", "di perpustakaan", atau "di rumah".
7. Jika tersedia, gunakan **beberapa kata dari daftar `context`** sebagai bagian dari cerita. Tidak harus semua digunakan.
8. Jika `context` kosong, tetap buat kisah sesuai dengan aturan.
9. Jangan gunakan dialog atau tanda kutip.
10. Jika cerita mengandung kata sifat yang menyakitkan atau unsur kriminal, ganti cerita dengan kisah sosial lain yang sesuai dan aman.
11. Hanya tampilkan satu baris dengan format:
   Story: ...
12. Gunakan kreativitasmu selama tetap sesuai aturan dan aman untuk anak-anak.


Format Output:
Story: ...

Contoh:
""",
    'four_person': r"""
Kamu adalah AI yang membuat *Kisah Sosial* (cerita pendek satu paragraf) yang ditujukan untuk anak-anak.

Petunjuk:
1. Tulis satu paragraf berupa kisah sosial yang **positif**, **mendidik**, dan **aman untuk anak-anak**.
2. Paragraf harus terdiri dari **5 sampai 10 kalimat sederhana**.
3. Gunakan **kata-kata yang mudah dimengerti oleh anak-anak**. Hindari kata sulit, bahasa gaul, atau ajakan langsung seperti "ayo", "yuk", atau "kamu harus".
4. Gunakan gaya bahasa yang **tenang, ramah, dan menyenangkan**.
5. Gunakan **empat nama orang**, yaitu <|PER|>, <|PER_1|>, <|PER_2|>, dan <|PER_3|>. Jangan gunakan nama lain selain itu.
6. Jangan gunakan nama tempat, nama jalan, atau nama khusus lain. Gantilah dengan kata umum atau baku seperti "di taman", "di perpustakaan", atau "di rumah".
7. Jika tersedia, gunakan **beberapa kata dari daftar `context`** sebagai bagian dari cerita. Tidak harus semua digunakan.
8. Jika `context` kosong, tetap buat kisah sesuai dengan aturan.
9. Jangan gunakan dialog atau tanda kutip.
10. Jika cerita mengandung kata sifat yang menyakitkan atau unsur kriminal, ganti cerita dengan kisah sosial lain yang sesuai dan aman.
11. Hanya tampilkan satu baris dengan format:
   Story: ...
12. Gunakan kreativitasmu selama tetap sesuai aturan dan aman untuk anak-anak.


Format Output:
Story: ...

Contoh:
""",
    'five_person': r"""
Kamu adalah AI yang membuat *Kisah Sosial* (cerita pendek satu paragraf) yang ditujukan untuk anak-anak.

Petunjuk:
1. Tulis satu paragraf berupa kisah sosial yang **positif**, **mendidik**, dan **aman untuk anak-anak**.
2. Paragraf harus terdiri dari **5 sampai 10 kalimat sederhana**.
3. Gunakan **kata-kata yang mudah dimengerti oleh anak-anak**. Hindari kata sulit, bahasa gaul, atau ajakan langsung seperti "ayo", "yuk", atau "kamu harus".
4. Gunakan gaya bahasa yang **tenang, ramah, dan menyenangkan**.
5. Gunakan **lima nama orang**, yaitu <|PER|>, <|PER_1|>, <|PER_2|>, <|PER_3|>, dan <|PER_4|>. Jangan gunakan nama lain selain itu.
6. Jangan gunakan nama tempat, nama jalan, atau nama khusus lain. Gantilah dengan kata umum atau baku seperti "di taman", "di perpustakaan", atau "di rumah".
7. Jika tersedia, gunakan **beberapa kata dari daftar `context`** sebagai bagian dari cerita. Tidak harus semua digunakan.
8. Jika `context` kosong, tetap buat kisah sesuai dengan aturan.
9. Jangan gunakan dialog atau tanda kutip.
10. Jika cerita mengandung kata sifat yang menyakitkan atau unsur kriminal, ganti cerita dengan kisah sosial lain yang sesuai dan aman.
11. Hanya tampilkan satu baris dengan format:
   Story: ...
12. Gunakan kreativitasmu selama tetap sesuai aturan dan aman untuk anak-anak.

Format Output:
Story: ...

Contoh:
""",
}


LIST_STORIES_PROMPT = {
    'one_person': [
        {
            'content': '<|PER|> bangun pagi dan langsung merapikan tempat tidurnya. Setelah itu, dia mencuci tangan sebelum sarapan bersama keluarganya. Ia merasa senang karena bisa membantu di rumah. Keluarganya tersenyum melihat sikapnya. Kegiatan pagi hari jadi lebih menyenangkan saat semuanya rapi dan bersih. <|PER|> belajar bahwa memulai hari dengan baik membuat semuanya terasa lebih mudah.'
        },
        {
            'content': 'Setiap hari, <|PER|> berjalan kaki ke sekolah. Di jalan, dia selalu menyapa orang-orang yang ditemuinya dengan senyum. Ia juga melambaikan tangan dan memberi salam. Orang-orang merasa senang melihat keramahan itu. Guru di sekolah bilang, <|PER|> adalah anak yang ramah. Saling menyapa membuat hari terasa lebih ceria dan hangat.'
        },
        {
            'content': 'Saat istirahat, <|PER|> melihat ada anak yang kesulitan membuka kotak makan. Tanpa diminta, dia langsung membantu. Anak itu tersenyum dan berterima kasih. Mereka lalu makan bersama sambil bercerita. <|PER|> merasa senang bisa membantu. Ia belajar bahwa menolong orang lain membuat hati jadi hangat.'
        },
        {
            'content': '<|PER|> bermain sendiri di taman dan menemukan dua ayunan kosong. Dia memilih satu dan membiarkan yang lain tetap kosong agar bisa digunakan anak lain. Ia bermain dengan sabar dan tersenyum saat melihat anak lain ikut bermain. <|PER|> belajar bahwa berbagi tempat membuat semua orang bisa bersenang-senang bersama.'
        },
        {
            'content': '<|PER|> sedang sedih karena mainannya rusak. Ibunya memeluknya dan berkata bahwa tidak apa-apa merasa sedih. Setelah itu, mereka mencoba memperbaikinya bersama-sama. Meskipun tidak berhasil, <|PER|> merasa lebih baik. Ia belajar bahwa perasaan sedih itu wajar dan bisa dihadapi bersama orang yang sayang padanya.'
        },
    ],
    'two_person': [
        {
            'content': '<|PER|> dan <|PER_1|> bangun pagi dan membantu merapikan tempat tidur masing-masing. Setelah itu, mereka mencuci tangan sebelum sarapan bersama keluarga. Mereka merasa senang karena bisa membantu di rumah. Keluarga tersenyum melihat sikap mereka. Kegiatan pagi jadi lebih menyenangkan saat semuanya rapi dan bersih. <|PER|> dan <|PER_1|> belajar bahwa memulai hari dengan baik membuat segalanya terasa lebih mudah.'
        },
        {
            'content': 'Setiap hari, <|PER|> dan <|PER_1|> berjalan kaki ke sekolah bersama. Di jalan, mereka saling bercerita dan tersenyum kepada orang-orang. Mereka juga memberi salam dengan ramah. Suasana pagi jadi lebih ceria. Guru di sekolah senang melihat sikap mereka. <|PER|> dan <|PER_1|> belajar bahwa bersikap ramah membuat semua orang merasa senang.'
        },
        {
            'content': 'Saat istirahat, <|PER|> melihat <|PER_1|> kesulitan membuka kotak makan. <|PER|> langsung membantu tanpa diminta. <|PER_1|> tersenyum dan berterima kasih. Mereka lalu makan bersama dan saling bercerita. <|PER|> merasa senang bisa membantu temannya. Mereka belajar bahwa saling tolong-menolong membuat hati menjadi hangat.'
        },
        {
            'content': '<|PER|> dan <|PER_1|> bermain bersama di taman. Mereka bergiliran naik ayunan dan tidak saling berebut. Mereka saling menyemangati dan tertawa bersama. Semua jadi terasa menyenangkan karena mereka bermain dengan adil. <|PER|> dan <|PER_1|> belajar bahwa sabar dan menghargai orang lain membuat bermain jadi lebih seru.'
        },
        {
            'content': '<|PER|> merasa sedih karena mainannya rusak. <|PER_1|> datang dan menghiburnya dengan mengajak bermain bersama. Mereka mencoba memperbaiki mainan itu, walaupun tidak berhasil. Namun, <|PER|> merasa lebih baik karena ditemani. <|PER_1|> juga merasa senang bisa membantu. Mereka belajar bahwa hadir untuk teman itu sangat berarti.'
        },
    ],
    'three_person': [
        {
            'content': '<|PER|>, <|PER_1|>, dan <|PER_2|> membantu membersihkan ruang kelas setelah pelajaran selesai. <|PER|> menyapu lantai, <|PER_1|> mengelap meja, dan <|PER_2|> membuang sampah ke tempatnya. Mereka bekerja sama dengan semangat dan saling membantu. Guru mereka tersenyum melihat kekompakan mereka. Setelah selesai, ruangan jadi bersih dan nyaman. <|PER|>, <|PER_1|>, dan <|PER_2|> belajar bahwa bekerja sama membuat pekerjaan terasa lebih ringan.'
        },
        {
            'content': '<|PER|>, <|PER_1|>, dan <|PER_2|> bermain petak umpet di halaman rumah. Mereka bergiliran menjadi penjaga dan tidak ada yang curang. Saat satu tertangkap, yang lain langsung tertawa bersama. Mereka senang karena bisa bermain dengan jujur dan adil. <|PER|> merasa permainan jadi lebih seru karena semua bermain baik. <|PER_1|> dan <|PER_2|> juga setuju. Mereka belajar bahwa bermain dengan aturan membuat semua senang.'
        },
        {
            'content': '<|PER|>, <|PER_1|>, dan <|PER_2|> membuat kerajinan dari kertas warna-warni. <|PER|> membuat bunga, <|PER_1|> membuat bintang, dan <|PER_2|> membuat hati. Mereka saling memuji hasil karya masing-masing. Setelah selesai, mereka menempelkan karya mereka di papan kelas. Guru mereka bangga dan memajangnya di dinding. Mereka belajar bahwa kreativitas dan saling menghargai membuat suasana jadi menyenangkan.'
        },
        {
            'content': '<|PER|> melihat <|PER_1|> kesulitan membawa buku, lalu mengajak <|PER_2|> untuk membantu bersama. Mereka membagi buku agar tidak berat. <|PER_1|> tersenyum dan mengucapkan terima kasih. Mereka lalu berjalan ke kelas sambil bercerita. <|PER|>, <|PER_1|>, dan <|PER_2|> merasa senang bisa saling tolong. Mereka belajar bahwa kebaikan kecil bisa membuat hari jadi lebih cerah.'
        },
        {
            'content': '<|PER|>, <|PER_1|>, dan <|PER_2|> pergi ke taman untuk piknik. Mereka membawa bekal masing-masing dan saling berbagi makanan. Saat ada sampah, mereka langsung membuangnya ke tempat sampah. Mereka menjaga kebersihan dan tidak membuang sampah sembarangan. Setelah makan, mereka bermain bersama dan tertawa riang. Mereka belajar bahwa menjaga lingkungan itu penting dan membuat semua orang senang.'
        },
    ],
    'four_person': [
        {
            'content': '<|PER|>, <|PER_1|>, <|PER_2|>, dan <|PER_3|> bermain bola di lapangan sekolah. Mereka membuat dua tim dan bermain dengan aturan yang adil. Saat satu tim mencetak gol, tim lain tetap memberi semangat. Mereka tidak marah atau kecewa, justru saling mendukung. Setelah selesai bermain, mereka berjabat tangan dan tersenyum. Mereka belajar bahwa bermain bersama itu lebih menyenangkan jika dilakukan dengan sportivitas dan saling menghargai.'
        },
        {
            'content': '<|PER|>, <|PER_1|>, <|PER_2|>, dan <|PER_3|> membuat poster kebersihan kelas. <|PER|> menggambar, <|PER_1|> mewarnai, <|PER_2|> menulis pesan, dan <|PER_3|> menempelkan hiasan. Mereka bekerja sama dengan baik dan saling memberi ide. Hasilnya sangat bagus dan dipajang di depan kelas. Guru mereka memberi pujian karena kekompakan mereka. Mereka belajar bahwa bekerja bersama dengan teman membuat hasil jadi lebih indah.'
        },
        {
            'content': '<|PER|>, <|PER_1|>, <|PER_2|>, dan <|PER_3|> mengikuti lomba estafet di sekolah. Mereka saling menyemangati dan berlari dengan semangat. Meskipun tidak menang, mereka tetap tersenyum dan memeluk satu sama lain. Mereka merasa bangga karena sudah berusaha dan bekerja sama. Guru mereka mengatakan bahwa semangat dan kebersamaan lebih penting dari juara. Mereka belajar bahwa berusaha bersama itu menyenangkan.'
        },
        {
            'content': '<|PER|>, <|PER_1|>, <|PER_2|>, dan <|PER_3|> duduk bersama di perpustakaan untuk membaca buku. <|PER|> membaca cerita hewan, <|PER_1|> memilih buku petualangan, <|PER_2|> suka buku bergambar, dan <|PER_3|> membaca ensiklopedia. Setelah membaca, mereka saling bercerita tentang isi bukunya. Mereka tertawa dan bertukar buku untuk dibaca lagi nanti. Mereka belajar bahwa membaca bersama teman membuat waktu lebih menyenangkan dan penuh ilmu.'
        },
        {
            'content': '<|PER|>, <|PER_1|>, <|PER_2|>, dan <|PER_3|> melihat ada sampah di halaman sekolah. Tanpa disuruh, mereka mengambilnya dan membuang ke tempat sampah. Mereka lalu saling mengingatkan untuk menjaga kebersihan. Setelah itu, mereka duduk di taman sambil menikmati udara segar. Guru mereka melihat dan mengucapkan terima kasih. Mereka belajar bahwa menjaga kebersihan adalah tanggung jawab bersama.'
        },
    ],
    'five_person': [
        {
            'content': '<|PER|>, <|PER_1|>, <|PER_2|>, <|PER_3|>, dan <|PER_4|> membuat taman kecil di halaman sekolah. <|PER|> membawa tanah, <|PER_1|> menanam bunga, <|PER_2|> menyiram air, <|PER_3|> membersihkan daun kering, dan <|PER_4|> membuat papan nama tanaman. Mereka bekerja sama dengan semangat dan saling membantu. Taman itu menjadi indah dan membuat suasana sekolah jadi lebih segar. Guru mereka bangga melihat hasilnya. Mereka belajar bahwa kerja sama dan cinta lingkungan membuat dunia jadi lebih baik.'
        },
        {
            'content': '<|PER|>, <|PER_1|>, <|PER_2|>, <|PER_3|>, dan <|PER_4|> bermain permainan papan di rumah <|PER|>. Mereka bergiliran melempar dadu dan mengikuti aturan dengan sabar. Saat salah satu menang, yang lain memberi selamat dengan senyum. Tidak ada yang marah atau kecewa. Mereka bermain dengan senang hati dan menghargai satu sama lain. Mereka belajar bahwa bermain bersama jadi lebih menyenangkan jika dilakukan dengan jujur dan ramah.'
        },
        {
            'content': '<|PER|>, <|PER_1|>, <|PER_2|>, <|PER_3|>, dan <|PER_4|> membuat kue bersama di dapur. <|PER|> mengaduk adonan, <|PER_1|> memecah telur, <|PER_2|> menakar bahan, <|PER_3|> menghias kue, dan <|PER_4|> menjaga oven. Mereka saling membantu dan tertawa bersama saat memasak. Ketika kue matang, mereka membaginya dengan adil. Rasanya enak dan mereka merasa bangga. Mereka belajar bahwa memasak bersama bisa menjadi pengalaman yang menyenangkan dan penuh kebersamaan.'
        },
        {
            'content': '<|PER|>, <|PER_1|>, <|PER_2|>, <|PER_3|>, dan <|PER_4|> mengikuti lomba kebersihan kelas. Mereka membagi tugas dengan baik, seperti menyapu, mengepel, merapikan meja, dan membersihkan papan tulis. Mereka bekerja dengan penuh semangat dan tidak ada yang bermalas-malasan. Kelas mereka menjadi sangat rapi dan bersih. Meskipun tidak menang, mereka tetap merasa senang. Mereka belajar bahwa usaha bersama lebih penting daripada hasil.'
        },
        {
            'content': '<|PER|>, <|PER_1|>, <|PER_2|>, <|PER_3|>, dan <|PER_4|> pergi ke perpustakaan bersama. Mereka memilih buku berbeda dan duduk membaca dalam diam. Setelah itu, mereka berbagi cerita tentang isi buku yang mereka baca. Mereka saling mendengarkan dan menghargai pendapat satu sama lain. Suasana jadi menyenangkan dan penuh ilmu. Mereka belajar bahwa membaca bersama teman bisa jadi pengalaman yang seru dan bermanfaat.'
        },
    ],
}
