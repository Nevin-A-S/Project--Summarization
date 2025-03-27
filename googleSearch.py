import time
import logging
import undetected_chromedriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def create_driver():
    """Create and configure a Selenium WebDriver instance."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Anti-detection measures
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return undetected_chromedriver.Chrome(service=service, options=chrome_options)

def scrape_google_results(query):
    """Scrape top 3 Google search results for the given query."""
    results = []
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            with create_driver() as driver:
                logger.info(f"Attempt {attempt + 1}: Searching Google for query: {query}")
                driver.get(f"https://www.google.com/search?q={query}")

                # Handle cookie consent
                try:
                    consent_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable(
                            (By.XPATH, "//button[contains(., 'I agree') or contains(., 'Accept all')]")
                        )
                    )
                    consent_button.click()
                    logger.info("Cookie consent accepted")
                except TimeoutException:
                    logger.info("No cookie consent dialog found")

                # Wait for presence of result-stats
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "result-stats"))
                )
                time.sleep(2)  # Allow dynamic content to load

                # Extract search results
                div_g_elements = driver.find_elements(By.CSS_SELECTOR, "div.g")
                logger.info(f"Found {len(div_g_elements)} div.g elements")

                for result in div_g_elements:
                    try:
                        title = result.find_element(By.CSS_SELECTOR, "h3").text.strip()
                        url = result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                        if url and url.startswith("http") and title:
                            results.append({"title": title, "url": url})
                            logger.info(f"Added result: {title} - {url}")
                            if len(results) >= 3:
                                break
                    except Exception as e:
                        logger.warning(f"Error extracting result: {e}")
                        continue

                if results:
                    break  # Exit if results are found
        except TimeoutException as e:
            logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
            if attempt == max_attempts - 1:
                logger.error("All attempts failed to retrieve search results")
        except WebDriverException as e:
            logger.error(f"WebDriver error: {e}")
            break

    return results[:3]

def clean_content(html_content):
    """Clean HTML content to extract readable text."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for unwanted in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
            unwanted.decompose()
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        text = main_content.get_text(strip=True) if main_content else soup.body.get_text(strip=True) if soup.body else ""
        return text
    except Exception as e:
        logger.error(f"Error cleaning content: {e}")
        return ""

def scrape_content(url):
    """Scrape and clean content from a given URL."""
    try:
        with create_driver() as driver:
            logger.info(f"Scraping content from: {url}")
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)  # Allow dynamic content
            return clean_content(driver.page_source)
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return ""

def google_content_cleaned(query):
    """Fetch and clean content from top 3 Google search results."""
    top_3_results = scrape_google_results(query)
    final_content = []
    for result in top_3_results:
        logger.info(f"Processing URL: {result['url']}")
        content = scrape_content(result["url"])
        if content:
            final_content.append(content)
    return ''.join(final_content)

if __name__ == "__main__":
    query = "matest machine learning news"
    result = google_content_cleaned(query)
    print(result)