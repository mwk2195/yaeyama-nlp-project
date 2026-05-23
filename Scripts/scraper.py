import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

dictionary_data = []

for page_num in range(1,911):
    print(f"Scraping page {page_num}")

    url = f"https://ninda.ninjal.ac.jp/s/hatoma/item?property%5B0%5D%5Bjoiner%5D=and&property%5B0%5D%5Bproperty%5D=312&property%5B0%5D%5Btype%5D=ex&page={page_num}&sort_by=hatoma%3Awd&sort_order=asc"

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to retrieve page {page_num}. Status code: {response.status_code}")
        break

    soup = BeautifulSoup(response.text, 'html.parser')

    entries = soup.find_all('div', class_='card-body')

    for entry in entries:
        try:
            title_tag = entry.find('h3', class_='card-title')

            phonetic = title_tag.contents[0].strip()
            katakana = title_tag.contents[2].strip()

            pos_tag = title_tag.find_next_sibling('div') 
            pos = pos_tag.text.strip().replace('[品詞] ', '')

            dictionary_data.append({
                'Phonemic':phonetic,
                'Katakana':katakana,
                'PartOfSpeech':pos
            })
        except (AttributeError, IndexError) as e:
            continue
    
    time.sleep(random.uniform(1.0,2.5))

df = pd.DataFrame(dictionary_data)
df.to_csv('hatoma_seed_corpus.csv', index=False, encoding='utf-8-sig')

print(f"Done! Scraped {len(df)} entries.")