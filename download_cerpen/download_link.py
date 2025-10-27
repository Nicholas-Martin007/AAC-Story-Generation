import json

from bs4 import BeautifulSoup

# Baca HTML
with open(
    'download_cerpen/cerpen digital _ Harian Kompas.html',
    'r',
    encoding='utf-8',
) as file:
    soup = BeautifulSoup(file, 'html.parser')

divs = soup.find_all(
    'div',
    class_='pb-6 mb-6 border-b border-grey-30 w-full',
)

# Ambil href <a> pertama
hrefs = []
for div in divs:
    a_tag = div.find('a')
    if a_tag and a_tag.get('href'):
        hrefs.append(a_tag['href'])


with open(
    'link_cerpen.txt',
    'w',
    encoding='utf-8',
) as f:
    json.dump(hrefs, f, indent=2)

print(hrefs)
