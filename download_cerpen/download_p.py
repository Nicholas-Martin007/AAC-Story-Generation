import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urlparse

with open("download_cerpen/link_cerpen.txt", "r") as f:
    urls = json.load(f)

target_classes = [
    "w-full max-w-[624px] px-4 md:px-0 md:mx-auto paragraph break-words mb-4 text-body font-lora rendered-component",
    "ksm-GMg ksm-2BC"
]

output = {}

for url in urls:
    print(f"Fetching: {url}")
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")

    slug = urlparse(url).path.split("/")[-1].split("?")[0]

    paragraphs = []
    for cls in target_classes:
        paragraphs += soup.find_all("p", class_=cls)

    if not paragraphs:
        print(f"❌ No paragraphs for: {slug} || {cls}")
        continue

    output[slug] = {
        "text": [p.get_text(strip=True) for p in paragraphs]
    }

with open("raw-text-cerpen.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
