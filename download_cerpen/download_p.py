# DOWNLOAD TEXT (STEP 2)

from typing import Iterator, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath('./'))
from utils.file_utils import *


def web_scrapping(
    urls: List[str],
) -> Iterator[List[str]]:
    """
    Ambil isi paragraf

    return Map(List)
    """

    target_class = [
        'w-full max-w-[624px] px-4 md:px-0 md:mx-auto paragraph break-words mb-4 text-body font-lora rendered-component',
        'ksm-GMg ksm-2BC',
    ]

    result = {}

    for url in tqdm(urls, desc='Web Scrapping'):
        response = requests.get(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0',
            },
        )
        soup = BeautifulSoup(response.text, 'html.parser')

        # ambil judul
        cerpen_name = (
            urlparse(url).path.split('/')[-1].split('?')[0]
        )

        p = []
        for c in target_class:
            p += soup.find_all('p', class_=c)

        if not p:
            continue

        result[cerpen_name] = {
            'text': [p.get_text(strip=True) for p in p]
        }

    return result


########################
urls = read_file(
    filename='download_cerpen/link_cerpen.txt',
)

scrapped_data = web_scrapping(urls=urls)

save_file(
    data=scrapped_data,
    save_filename='raw-text-cerpen.json',
    ensure_ascii=False,
)
