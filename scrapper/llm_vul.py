import requests
import re
import json
import time
import os
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
DB_FILE = 'promptfoo_full_export.json'
BASE_URL = "https://www.promptfoo.dev"
INDEX_URL = "https://www.promptfoo.dev/lm-security-db/"

def get_all_urls():
    """Extracts all vulnerability URLs from the main database page index."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
    print("Fetching main database index...")
    response = requests.get(INDEX_URL, headers=headers)
    # Regex to find the slugs in the Next.js data stream
    pattern = r'\\"slug\\":\\"(?P<slug>[^\\"]+)\\"'
    slugs = re.findall(pattern, response.text)
    unique_urls = list(set([f"{BASE_URL}/lm-security-db/vuln/{s}" for s in slugs]))
    print(f"Found {len(unique_urls)} total unique vulnerabilities.")
    return unique_urls

def scrape_page_details(session, url):
    """Scrapes the 'prose' content from an individual vulnerability page."""
    try:
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        prose_div = soup.select_one("div.prose")
        title = soup.find('h1').text.strip() if soup.find('h1') else "N/A"
        
        if not prose_div: return None

        data = {"title": title, "url": url, "details": {}}
        current_header = "General"
        
        for element in prose_div.find_all(['p', 'ul', 'li']):
            strong = element.find('strong')
            if strong and ":" in strong.text:
                current_header = strong.text.replace(':', '').strip()
                text = element.get_text().replace(strong.text, '').strip()
                data["details"][current_header] = text
            else:
                existing = data["details"].get(current_header, "")
                data["details"][current_header] = f"{existing}\n{element.get_text().strip()}".strip()
        return data
    except Exception as e:
        print(f"Error on {url}: {e}")
        return None

def main():
    # 1. Initialize the file if it doesn't exist
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)

    # 2. Get list of all URLs
    all_urls = get_all_urls()

    # 3. Load existing progress to avoid scraping the same page twice
    with open(DB_FILE, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    
    scraped_urls = {item['url'] for item in existing_data}
    print(f"Already have {len(scraped_urls)} entries. Resuming...")

    # 4. Setup Session
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"})

    # 5. Start Scraping
    for i, url in enumerate(all_urls):
        if url in scraped_urls:
            continue
        
        print(f"[{i+1}/{len(all_urls)}] Scraping: {url}")
        detail = scrape_page_details(session, url)
        
        if detail:
            # Append to the file immediately
            existing_data.append(detail)
            with open(DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
        
        # Polite delay
        time.sleep(0.5)

    print(f"\nFinished! Total records in {DB_FILE}: {len(existing_data)}")

if __name__ == "__main__":
    main()