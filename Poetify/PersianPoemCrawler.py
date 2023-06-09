import requests
from lxml import html
from tqdm import tqdm

def crawl_poems(url, filename):
    page = requests.get(url)
    tree = html.fromstring(page.content)
    
    poems = tree.xpath('//div[@class="b"]/div/p/text()')
    
    with open(filename, 'a', encoding="utf-8") as f:
        for poem in poems:
            formatted_poem = poem.replace('\u200c', ' ')
            f.write(f'|{formatted_poem}\n')

def crawl_khayyam_poems():
    for i in tqdm(range(1, 179)):
        url = f'https://ganjoor.net/khayyam/robaee/sh{i}/'
        filename = 'khayaam.txt'
        crawl_poems(url, filename)

def crawl_moulavi_poems():
    for i in tqdm(range(1, 3231)):
        url = f'https://ganjoor.net/moulavi/shams/ghazalsh/sh{i}/'
        filename = 'moulavi.txt'
        crawl_poems(url, filename)

def merge_files(file1, file2, output_file):
    with open(file1, 'r', encoding="utf-8") as f1, open(file2, 'r', encoding="utf-8") as f2, open(output_file, 'w', encoding="utf-8") as out_file:
        data1 = f1.read()
        data2 = f2.read()
        out_file.write(data1)
        out_file.write(data2)

crawl_khayyam_poems()
crawl_moulavi_poems()

# Call the merge_files function after the completion of crawl_khayyam_poems and crawl_moulavi_poems
file1 = 'khayaam.txt'
file2 = 'moulavi.txt'
output_file = 'merged_poems.txt'
merge_files(file1, file2, output_file)
