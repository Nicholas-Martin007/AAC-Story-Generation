# DOWNLOAD LINK (STEP 1)

import json
from typing import List

from bs4 import BeautifulSoup


def get_soup() -> BeautifulSoup:
    """
    Parse HTML

    return HTML data
    """
    with open(
        'download_cerpen/cerpen digital _ Harian Kompas.html',
        'r',
        encoding='utf-8',
    ) as file:
        return BeautifulSoup(file, 'html.parser')


def get_href(soup: BeautifulSoup) -> List[str]:
    divs = soup.find_all(
        'div',
        class_='pb-6 mb-6 border-b border-grey-30 w-full',
    )

    hrefs = []
    for div in divs:
        a_tag = div.find('a')
        if a_tag and a_tag.get('href'):
            hrefs.append(a_tag['href'])

    print(hrefs)

    return hrefs


def save_file(
    data: List[str],
    save_filename: str,
    ensure_ascii: bool = True,
) -> None:
    with open(
        save_filename,
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=ensure_ascii,
        )


#########################
soup = get_soup()

hrefs = get_href(soup=soup)

save_file(data=hrefs, save_filename='link_cerpen.txt')
