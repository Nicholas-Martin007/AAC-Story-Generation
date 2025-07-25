# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# import time

# options = Options()
# options.add_argument("--headless")
# driver = webdriver.Chrome(options=options)

# url = "https://www.kompas.id/label/cerpen-digital"
# driver.get(url)

# wait = WebDriverWait(driver, 10)

# button_class = "_base_x9yc9_2 _bordr_x9yc9_78 _contentBold_x9yc9_123 _txtCapitalize_x9yc9_110 _btnBase_x9yc9_38 _round1_x9yc9_90 font-sans"

# for _ in range(50):
#     try:
#         button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "font-sans")))
#         ActionChains(driver).move_to_element(button).click().perform()
#         time.sleep(1.5)
#     except:
#         print("No more button or failed to click.")
#         break

# target_class = "_base_19dt3_2 _borderNil_19dt3_18 _borderTransparent_19dt3_22 block clearfix text-grey-60"
# divs = driver.find_elements(By.CLASS_NAME, "text-grey-60")

# links = []
# for div in divs:
#     try:
#         a_tag = div.find_element(By.TAG_NAME, "a")
#         links.append(a_tag.get_attribute("href"))
#     except:
#         continue

# # Output links
# for link in links:
#     print(link)

# driver.quit()

from bs4 import BeautifulSoup

# Baca file HTML
with open(
    "download_cerpen/cerpen digital _ Harian Kompas.html", "r", encoding="utf-8"
) as file:
    soup = BeautifulSoup(file, "html.parser")

# Cari semua div dengan class target
divs = soup.find_all("div", class_="pb-6 mb-6 border-b border-grey-30 w-full")

# Ambil href dari <a> pertama di dalam setiap div
hrefs = []
for div in divs:
    a_tag = div.find("a")
    if a_tag and a_tag.get("href"):
        hrefs.append(a_tag["href"])


import json

with open(f"link_cerpen.txt", "w", encoding="utf-8") as f:
    json.dump(hrefs, f, indent=2)

print(hrefs)
