from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup 

def scrape_google_results(query):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get("https://www.google.com")
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "search")))
        results = driver.find_elements(By.CSS_SELECTOR, "div.g")[:3]
        return [
            {
                "title": result.find_element(By.CSS_SELECTOR, "h3").text,
                "url": result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
            }
            for result in results
        ]
    finally:
        driver.quit()

def clean_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unwanted elements
    for unwanted in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
        unwanted.decompose()
    
    # Try to find the main content
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
    
    if main_content:
        return main_content.get_text(strip=True)
    else:
        # If we can't find a clear main content, just return the body text
        return soup.body.get_text(strip=True)

def scrape_content(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        html_content = driver.page_source
        return clean_content(html_content)
    finally:
        driver.quit()


def google_content_cleaned(query):
    top_3_results = scrape_google_results(query)
    final = []
    for result in top_3_results:
        # print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        content = scrape_content(result['url'])
        final.append(content)
    
    return ''.join(final)

if __name__ == '__main__':
    query = "matest machine learning news"
    print(google_content_cleaned(query))