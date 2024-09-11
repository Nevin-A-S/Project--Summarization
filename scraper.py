import requests
from bs4 import BeautifulSoup
import re

def scraper(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find(id="firstHeading")

    paragraphs = soup.find_all("p")
    content = []
    for para in paragraphs:
        content.append(para.get_text())
    
    content_cleaned = [re.sub(r'\[\d+\]', '', para) for para in content]
       

    return content_cleaned