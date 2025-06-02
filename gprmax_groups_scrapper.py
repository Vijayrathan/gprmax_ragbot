import requests
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin, urlparse
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoogleGroupsScraper:
    def __init__(self, base_url="https://groups.google.com/g/gprmax", headless=True):
        self.base_url = base_url
        self.conversations = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Set up Selenium WebDriver
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {e}")
            logger.info("Please make sure ChromeDriver is installed and in PATH")
            raise

    def get_conversation_links(self):
        """Extract all conversation links from the main group page using pagination"""
        all_conversation_links = set()  # Use set to avoid duplicates
        page_count = 0
        max_pages = 50  # Safety limit
        
        try:
            logger.info("Loading main group page...")
            self.driver.get(self.base_url)
            
            # Wait for the page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for initial content
            time.sleep(3)
            
            logger.info("Starting pagination to collect all conversation links...")
            
            while page_count < max_pages:
                page_count += 1
                logger.info(f"Scraping page {page_count}...")
                
                # Get conversation links from current page
                conversation_links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/c/"]')
                current_page_links = set()
                
                for link in conversation_links:
                    href = link.get_attribute('href')
                    if href and '/c/' in href:
                        current_page_links.add(href)
                
                logger.info(f"  Found {len(current_page_links)} links on page {page_count}")
                
                # Add to our collection
                before_count = len(all_conversation_links)
                all_conversation_links.update(current_page_links)
                after_count = len(all_conversation_links)
                new_links = after_count - before_count
                
                logger.info(f"  Added {new_links} new links (total: {after_count})")
                
                # Look for next page button
                next_buttons = self.driver.find_elements(By.CSS_SELECTOR, '[aria-label="Next page"]')
                
                if not next_buttons:
                    logger.info("No 'Next page' button found. Trying alternative selectors...")
                    # Try alternative selectors for next button
                    alternative_selectors = [
                        '[aria-label*="next"]',
                        '[aria-label*="Next"]',
                        'button[aria-label*="next"]',
                        'button[aria-label*="Next"]',
                        '[role="button"][aria-label*="next"]',
                        '[role="button"][aria-label*="Next"]'
                    ]
                    
                    for selector in alternative_selectors:
                        next_buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if next_buttons:
                            logger.info(f"Found next button with selector: {selector}")
                            break
                
                if not next_buttons:
                    logger.info("No next page button found. Reached end of pagination.")
                    break
                
                # Check if the next button is enabled/clickable
                next_button = next_buttons[0]  # Take the first one
                
                # Check if button is disabled
                disabled = next_button.get_attribute('disabled')
                aria_disabled = next_button.get_attribute('aria-disabled')
                
                if disabled or aria_disabled == 'true':
                    logger.info("Next page button is disabled. Reached end of pagination.")
                    break
                
                # Try to click the next button
                try:
                    logger.info("Clicking next page button...")
                    
                    # Scroll to the button first
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                    time.sleep(1)
                    
                    # Try clicking
                    next_button.click()
                    
                    # Wait for page to load
                    time.sleep(3)
                    
                    # Verify we moved to a new page by checking if content changed
                    new_conversation_links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/c/"]')
                    new_page_links = set()
                    for link in new_conversation_links:
                        href = link.get_attribute('href')
                        if href and '/c/' in href:
                            new_page_links.add(href)
                    
                    # If we have the exact same links, we might not have moved to a new page
                    if new_page_links == current_page_links:
                        logger.warning("Same links found after clicking next. Might be at the end.")
                        break
                    
                except Exception as e:
                    logger.error(f"Error clicking next button: {e}")
                    # Try JavaScript click as fallback
                    try:
                        logger.info("Trying JavaScript click...")
                        self.driver.execute_script("arguments[0].click();", next_button)
                        time.sleep(3)
                    except Exception as e2:
                        logger.error(f"JavaScript click also failed: {e2}")
                        break
            
            # Convert set to list
            conversation_links = list(all_conversation_links)
            
        except Exception as e:
            logger.error(f"Error getting conversation links: {e}")
            conversation_links = []
        
        logger.info(f"Found {len(conversation_links)} unique conversation links across {page_count} pages")
        return conversation_links

    def extract_conversation_content(self, url):
        """Extract content from a single conversation"""
        conversation_data = {
            'url': url,
            'title': '',
            'original_post': {
                'author': '',
                'date': '',
                'content': ''
            },
            'responses': []
        }
        
        try:
            logger.info(f"Scraping conversation: {url}")
            self.driver.get(url)
            
            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            time.sleep(3)
            
            # Extract title
            title_selectors = [
                'h1.KPwZRb.gKR4Fb',
                'html-blob',
                '.KPwZRb',
                'h1'
            ]
            
            for selector in title_selectors:
                try:
                    title_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    conversation_data['title'] = title_element.text.strip()
                    if conversation_data['title']:
                        break
                except:
                    continue
            
            # Extract messages using the correct structure
            message_sections = self.driver.find_elements(By.CSS_SELECTOR, 'section.BkrUxb')
            
            for i, section in enumerate(message_sections):
                try:
                    # Skip deleted messages
                    if 'Message has been deleted' in section.text:
                        continue
                    
                    # Extract author
                    author = ''
                    author_selectors = [
                        'h3.s1f8Zd',
                        '.s1f8Zd',
                        '.LgTNRd h3'
                    ]
                    
                    for selector in author_selectors:
                        try:
                            author_element = section.find_element(By.CSS_SELECTOR, selector)
                            author = author_element.text.strip()
                            if author:
                                break
                        except:
                            continue
                    
                    # Extract date
                    date = ''
                    date_selectors = [
                        '.zX2W9c',
                        '.ELCJ4d .zX2W9c',
                        '.Rrziwd'
                    ]
                    
                    for selector in date_selectors:
                        try:
                            date_element = section.find_element(By.CSS_SELECTOR, selector)
                            date = date_element.text.strip()
                            if date:
                                break
                        except:
                            continue
                    
                    # Extract content using the correct selector
                    content = ''
                    content_selectors = [
                        '.ptW7te[jsname="yjbGtf"]',
                        '.ptW7te',
                        'div[dir="ltr"]'
                    ]
                    
                    for selector in content_selectors:
                        try:
                            content_element = section.find_element(By.CSS_SELECTOR, selector)
                            content = content_element.text.strip()
                            if content and len(content) > 10:  # Ensure we have substantial content
                                break
                        except:
                            continue
                    
                    # Create message data
                    message_data = {
                        'author': author,
                        'date': date,
                        'content': content
                    }
                    
                    # Add to appropriate section
                    if i == 0:  # First message is original post
                        conversation_data['original_post'] = message_data
                    else:  # Subsequent messages are responses
                        conversation_data['responses'].append(message_data)
                    
                    logger.info(f"Extracted message from {author}: {content[:100]}...")
                    
                except Exception as e:
                    logger.warning(f"Error extracting message {i}: {str(e)}")
                    continue
            
            return conversation_data
            
        except Exception as e:
            logger.error(f"Error extracting conversation content from {url}: {str(e)}")
            return conversation_data

    def scrape_all_conversations(self):
        """Main method to scrape all conversations"""
        try:
            # Get all conversation links
            conversation_links = self.get_conversation_links()
            
            if not conversation_links:
                logger.warning("No conversation links found. The page structure might have changed.")
                return []
            
            logger.info(f"Starting to scrape {len(conversation_links)} conversations...")
            
            # Scrape each conversation
            for i, link in enumerate(conversation_links):
                try:
                    logger.info(f"Processing conversation {i+1}/{len(conversation_links)}")
                    conversation_data = self.extract_conversation_content(link)
                    
                    if conversation_data['title'] or conversation_data['original_post']['content']:
                        self.conversations.append(conversation_data)
                        logger.info(f"Successfully scraped: {conversation_data['title']}")
                    else:
                        logger.warning(f"No content extracted from: {link}")
                    
                    # Save progress every 10 conversations
                    if (i + 1) % 10 == 0:
                        self.save_to_json(f"gprmax_conversations_progress_{i+1}.json")
                        logger.info(f"Progress saved: {i+1}/{len(conversation_links)} conversations")
                    
                    # Add delay to be respectful
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error processing conversation {link}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in scrape_all_conversations: {e}")
        
        return self.conversations

    def save_to_json(self, filename="gprmax_conversations.json"):
        """Save scraped conversations to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, indent=2, ensure_ascii=False)
            logger.info(f"Data saved to {filename}")
            logger.info(f"Total conversations scraped: {len(self.conversations)}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")

    def close(self):
        """Close the WebDriver"""
        if hasattr(self, 'driver'):
            self.driver.quit()

def main():
    scraper = None
    try:
        # Initialize scraper
        scraper = GoogleGroupsScraper(headless=True)  # Set to False to see browser
        
        # Scrape all conversations
        conversations = scraper.scrape_all_conversations()
        
        # Save to JSON
        scraper.save_to_json("gprmax_conversations.json")
        
        # Print summary
        print(f"\nScraping completed!")
        print(f"Total conversations scraped: {len(conversations)}")
        
        if conversations:
            print(f"\nSample conversation titles:")
            for i, conv in enumerate(conversations[:5]):  # Show first 5 titles
                print(f"{i+1}. {conv['title']}")
        
    except Exception as e:
        logger.error(f"Main execution error: {e}")
    finally:
        if scraper:
            scraper.close()

if __name__ == "__main__":
    main()