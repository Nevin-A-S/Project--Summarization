import requests
from bs4 import BeautifulSoup
import re
import logging
import time
from urllib.parse import urlparse
from fake_useragent import UserAgent
from requests.exceptions import RequestException
import html2text
from readability import readability  # Changed import
import trafilatura
import concurrent.futures
import extruct
from w3lib.html import get_base_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_headers():
    """Generate random user agent headers to avoid blocking"""
    try:
        ua = UserAgent()
        return {
            'User-Agent': ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    except Exception as e:
        logger.warning(f"Failed to generate random user agent: {e}")
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }

def handle_request(url, max_retries=3, backoff_factor=0.3):
    """Make HTTP request with retries and exponential backoff"""
    session = requests.Session()
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = session.get(url, headers=get_headers(), timeout=30)
            response.raise_for_status()
            return response
        except RequestException as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                return None
            
            wait_time = backoff_factor * (2 ** (retry_count - 1))
            logger.warning(f"Request failed: {e}. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)

def extract_with_bs4(html):
    """Extract content using BeautifulSoup"""
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script, style elements and comments
    for element in soup(["script", "style"]):
        element.decompose()
    for comment in soup.find_all(text=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
        comment.extract()
    
    # Get main title
    title = None
    if soup.find(id="firstHeading"):
        title = soup.find(id="firstHeading").get_text()
    elif soup.title:
        title = soup.title.get_text()
    elif soup.find("h1"):
        title = soup.find("h1").get_text()
    
    # Extract paragraphs
    paragraphs = soup.find_all("p")
    content = []
    
    # Add title if found
    if title:
        content.append(f"Title: {title}")
    
    # Extract main content
    main_content = soup.find(["main", "article", "div", "section"], 
                          class_=re.compile(r"content|main|article|post"))
    
    if main_content:
        paragraphs = main_content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])
    
    # If no main content found, use all paragraphs
    if not paragraphs:
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])
    
    for para in paragraphs:
        text = para.get_text().strip()
        if text:  # Only add non-empty paragraphs
            content.append(text)
    
    # Clean content (remove citation numbers, etc.)
    content_cleaned = [re.sub(r'\[\d+\]', '', para) for para in content]
    content_cleaned = [para.strip() for para in content_cleaned if para.strip()]
    
    return content_cleaned

def extract_with_readability(html):
    """Extract content using the readability algorithm"""
    try:
        doc = readability.Document(html)  # Changed to use correct class
        readable_article = doc.summary()
        readable_title = doc.title()
        
        # Convert HTML to text
        h = html2text.HTML2Text()
        h.ignore_links = False
        text = h.handle(readable_article)
        
        # Split into paragraphs and clean
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = [re.sub(r'\[\d+\]', '', p.strip()) for p in paragraphs]
        cleaned_paragraphs = [p for p in cleaned_paragraphs if p.strip()]
        
        # Add title if available
        if readable_title:
            cleaned_paragraphs.insert(0, f"Title: {readable_title}")
            
        return cleaned_paragraphs
    except Exception as e:
        logger.warning(f"Readability extraction failed: {e}")
        return []

def extract_with_trafilatura(html, url):
    """Extract content using trafilatura library"""
    try:
        extracted_text = trafilatura.extract(html, url=url, include_tables=True)
        if extracted_text:
            paragraphs = extracted_text.split('\n\n')
            cleaned = [p.strip() for p in paragraphs if p.strip()]
            return cleaned
        return []
    except Exception as e:
        logger.warning(f"Trafilatura extraction failed: {e}")
        return []

def extract_metadata(html, url):
    """Extract metadata from webpage"""
    try:
        base_url = get_base_url(html, url)
        metadata = extruct.extract(html, base_url=base_url, 
                                  syntaxes=['json-ld', 'microdata', 'opengraph'])
        if metadata:
            return metadata
        return {}
    except Exception as e:
        logger.warning(f"Metadata extraction failed: {e}")
        return {}

def consolidate_results(results_list):
    """Combine and deduplicate results from different extraction methods"""
    # Flatten all results
    all_paragraphs = []
    for result in results_list:
        if result:
            all_paragraphs.extend(result)
    
    # Remove duplicates while preserving order
    seen = set()
    consolidated = []
    for item in all_paragraphs:
        normalized = re.sub(r'\s+', ' ', item).strip()
        if normalized and normalized not in seen and len(normalized) > 10:
            seen.add(normalized)
            consolidated.append(item)
    
    return consolidated

def is_valid_url(url):
    """Check if URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def scraper(url):
    """
    Main scraper function that extracts content from a website
    
    Args:
        url (str): URL of the website to scrape
        
    Returns:
        list: List of text paragraphs from the website
    """
    if not is_valid_url(url):
        logger.error(f"Invalid URL: {url}")
        return []
    
    # Fetch the webpage
    response = handle_request(url)
    if not response:
        return []
    
    html = response.text
    domain = urlparse(url).netloc
    
    logger.info(f"Scraping content from {domain}")
    
    # Use multiple extraction methods in parallel
    extraction_methods = [
        (extract_with_bs4, [html]),
        (extract_with_readability, [html]),
        (extract_with_trafilatura, [html, url])
    ]
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_method = {
            executor.submit(method, *args): method.__name__ 
            for method, args in extraction_methods
        }
        
        for future in concurrent.futures.as_completed(future_to_method):
            method_name = future_to_method[future]
            try:
                result = future.result()
                if result:
                    logger.info(f"Extracted {len(result)} paragraphs with {method_name}")
                    results.append(result)
                else:
                    logger.warning(f"No content extracted with {method_name}")
            except Exception as e:
                logger.error(f"{method_name} extraction failed: {e}")
    
    # Extract metadata (optional)
    try:
        metadata = extract_metadata(html, url)
        if metadata:
            logger.info(f"Extracted metadata: {len(metadata)} items")
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
    
    # Consolidate results from different methods
    final_content = consolidate_results(results)
    
    if not final_content:
        logger.warning(f"No content extracted from {url}")
        return []
    
    logger.info(f"Successfully extracted {len(final_content)} paragraphs from {url}")
    return final_content


if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/Web_scraping"
    content = scraper(test_url)
    print("*"*80)
    print(f"Scraped content from {test_url}")
    print(content)
    print("*"*80)
    print(f"Extracted {len(content)} paragraphs")
    for i, para in enumerate(content[:5]):
        print(f"{i+1}. {para[:100]}...")