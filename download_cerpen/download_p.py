import json
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

with open(
    'download_cerpen/link_cerpen.txt',
    'r',
) as f:
    urls = json.load(f)

html_class = [
    'w-full max-w-[624px] px-4 md:px-0 md:mx-auto paragraph break-words mb-4 text-body font-lora rendered-component',
    'ksm-GMg ksm-2BC',
]

result = {}

# web scrapping
for url in urls:
    # ambil text html
    response = requests.get(
        url,
        headers={'User-Agent': 'Mozilla/5.0'},
    )
    soup = BeautifulSoup(response.text, 'html.parser')

    # buat ambil judul
    cerpen_name = urlparse(url).path.split('/')[-1].split('?')[0]

    p = []
    for c in html_class:
        p += soup.find_all('p', class_=c)

    if not p:
        continue

    # insert
    result[cerpen_name] = {
        'text': [p.get_text(strip=True) for p in p]
    }

# save
with open(
    'raw-text-cerpen.json',
    'w',
    encoding='utf-8',
) as f:
    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2,
    )


# kasih deskripsi
