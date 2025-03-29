import logging
import time
import re
import random
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from urllib.parse import quote_plus, urlparse, parse_qs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("scraper.log")]
)
logger = logging.getLogger(__name__)

def get_random_user_agent() -> str:
    """Return a random user agent string."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/116.0",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/116.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    ]
    return random.choice(user_agents)

def create_session() -> requests.Session:
    """Create and configure a requests session with appropriate headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0"
    })
    return session

def get_search_results(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    Get search results for a query using multiple methods.
    
    Args:
        query: Search query string
        num_results: Number of results to return
        
    Returns:
        List of dictionaries with title and url keys
    """
    # Try multiple search sources
    methods = [
        get_results_from_ddg,
        get_results_from_bing
    ]
    
    for method in methods:
        try:
            logger.info(f"Trying search method: {method.__name__}")
            results = method(query, num_results)
            if results:
                return results
            time.sleep(1)  # Delay between attempts
        except Exception as e:
            logger.error(f"Error in {method.__name__}: {e}")
    
    # If all methods fail, return dummy results for testing
    logger.warning("All search methods failed, returning dummy results")
    return create_dummy_results(query, num_results)

def get_results_from_ddg(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    Get search results from DuckDuckGo.
    
    Args:
        query: Search query string
        num_results: Number of results to return
        
    Returns:
        List of dictionaries with title and url keys
    """
    results = []
    try:
        session = create_session()
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        logger.info(f"Searching DuckDuckGo for: {query}")
        response = session.get(url, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"DuckDuckGo returned status code: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract results
        for result in soup.select('.result'):
            title_elem = result.select_one('.result__title')
            link_elem = result.select_one('.result__url')
            
            if title_elem and link_elem:
                title = title_elem.get_text(strip=True)
                href = link_elem.get('href')
                
                if href and title:
                    # Extract actual URL from DuckDuckGo redirect
                    parsed_url = urlparse(href)
                    if parsed_url.query:
                        query_params = parse_qs(parsed_url.query)
                        if 'uddg' in query_params:
                            url = query_params['uddg'][0]
                        else:
                            url = href
                    else:
                        url = href
                        
                    if url.startswith('http'):
                        results.append({"title": title, "url": url})
                        logger.info(f"Found result: {title} - {url}")
                        
                        if len(results) >= num_results:
                            break
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching DuckDuckGo: {e}")
        return []

def get_results_from_bing(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    Get search results from Bing.
    
    Args:
        query: Search query string
        num_results: Number of results to return
        
    Returns:
        List of dictionaries with title and url keys
    """
    results = []
    try:
        session = create_session()
        url = f"https://www.bing.com/search?q={quote_plus(query)}"
        
        logger.info(f"Searching Bing for: {query}")
        response = session.get(url, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"Bing returned status code: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract results - Bing's structure
        for result in soup.select('.b_algo'):
            title_elem = result.select_one('h2')
            link_elem = result.select_one('a')
            
            if title_elem and link_elem:
                title = title_elem.get_text(strip=True)
                url = link_elem.get('href')
                
                if url and url.startswith('http') and title:
                    results.append({"title": title, "url": url})
                    logger.info(f"Found result: {title} - {url}")
                    
                    if len(results) >= num_results:
                        break
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching Bing: {e}")
        return []

def create_dummy_results(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    Create dummy search results for testing when all search methods fail.
    
    Args:
        query: The search query
        num_results: Number of results to generate
        
    Returns:
        List of dummy result dictionaries
    """
    domains = ["example.com", "testsite.org", "dummydata.net", "samplecontent.edu", "mockup.info"]
    topics = query.split()
    
    results = []
    for i in range(min(num_results, 5)):
        domain = domains[i % len(domains)]
        topic = topics[i % len(topics)] if topics else "topic"
        results.append({
            "title": f"Sample Result {i+1} for {query}",
            "url": f"https://www.{domain}/{topic}-article-{i+1}"
        })
        logger.info(f"Created dummy result: {results[-1]['title']} - {results[-1]['url']}")
    
    return results

def scrape_content(url: str) -> str:
    """
    Scrape and clean content from a URL.
    
    Args:
        url: URL to scrape
        
    Returns:
        Cleaned text content
    """
    try:
        session = create_session()
        logger.info(f"Scraping content from: {url}")
        
        response = session.get(url, timeout=15)
        
        if response.status_code != 200:
            logger.warning(f"Failed to fetch content: HTTP {response.status_code}")
            return f"[Unable to fetch content from {url} - Status code {response.status_code}]"
        
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type and 'application/xhtml+xml' not in content_type:
            logger.warning(f"Not an HTML page: {content_type}")
            return f"[Content type not supported: {content_type}]"
        
        # Extract clean content
        return extract_main_content(response.text, url)
        
    except requests.RequestException as e:
        logger.error(f"Request error for {url}: {e}")
        return f"[Error fetching content: {str(e)}]"
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return f"[Error processing content: {str(e)}]"

def extract_main_content(html: str, url: str) -> str:
    """
    Extract main content from HTML.
    
    Args:
        html: HTML content
        url: Source URL for logging
        
    Returns:
        Extracted main content as text
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for tag in ('script', 'style', 'nav', 'header', 'footer', 'aside'):
            for element in soup.find_all(tag):
                element.decompose()
        
        # Try to identify main content
        main_content = None
        
        # Try different strategies to find main content
        for selector in [
            'main', 'article', 'div#content', 'div#main', 'div.content', 'div.main', 
            'div.article', '#article', '.post-content', '.entry-content'
        ]:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 100:
                main_content = content
                logger.info(f"Found main content using selector: {selector}")
                break
        
        # If no main content found with selectors, try to find the div with most text
        if not main_content:
            logger.info("Using text density algorithm to find main content")
            text_blocks = []
            for div in soup.find_all('div'):
                text = div.get_text(strip=True)
                if len(text) > 200:  # Minimum text length to consider
                    text_blocks.append((div, len(text)))
            
            if text_blocks:
                # Sort by text length (descending)
                text_blocks.sort(key=lambda x: x[1], reverse=True)
                main_content = text_blocks[0][0]
        
        # If still no main content, use body
        if not main_content:
            logger.info("Using body as main content")
            main_content = soup.body or soup
        
        # Extract text
        text = main_content.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines
        
        # Limit length
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "... [content truncated]"
        
        return text
        
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return f"[Error extracting content: {str(e)}]"

def google_content_cleaned(query: str) -> str:
    """
    Fetch and clean content from top search results.
    
    Args:
        query: Search query string
        
    Returns:
        Combined content from search results
    """
    try:
        # Get search results
        results = get_search_results(query)
        
        if not results:
            logger.warning(f"No results found for query: {query}")
            return f"No search results found for: {query}"
        
        # Scrape content from each result
        all_content = []
        
        for i, result in enumerate(results):
            try:
                title = result["title"]
                url = result["url"]
                
                logger.info(f"Processing result {i+1}/{len(results)}: {title}")
                
                # Add delay between requests
                if i > 0:
                    time.sleep(random.uniform(1, 3))
                
                # Get content
                content = scrape_content(url)
                
                if content:
                    # Format the content with source information
                    source_header = f"SOURCE {i+1}: {title}"
                    source_url = f"URL: {url}"
                    separator = "-" * 80
                    
                    formatted_content = f"{separator}\n{source_header}\n{source_url}\n{separator}\n\n{content}\n\n"
                    all_content.append(formatted_content)
                    
                    logger.info(f"Successfully extracted content from {url}")
                else:
                    logger.warning(f"No content extracted from {url}")
                    all_content.append(f"{separator}\n{source_header}\n{source_url}\n{separator}\n\n[No content could be extracted]\n\n")
            
            except Exception as e:
                logger.error(f"Error processing result {i+1}: {e}")
                continue
            
        if all_content:
            return "\n".join(all_content)
        else:
            return f"Failed to extract content for query: {query}"
            
    except Exception as e:
        logger.error(f"Error in google_content_cleaned: {e}")
        return f"Error processing search query: {str(e)}"


if __name__ == "__main__":
    query = "latest machine learning news"
    result = google_content_cleaned(query)
    print(f"Content length: {len(result)} characters")
    print("\nSample of content:")
    print(result[:500] + "..." if len(result) > 500 else result)