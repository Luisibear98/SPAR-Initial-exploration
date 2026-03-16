import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from datetime import datetime

BASE_URL = "https://www.greaterwrong.com"
TAG_URL = BASE_URL + "/tag/deceptive-alignment?sort=new"
OUTPUT_FILE = "greaterwrong_deceptive_alignment_full.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# --------------------------
# Utilities
# --------------------------

def get_soup(url):
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def load_existing_urls():
    if not os.path.exists(OUTPUT_FILE):
        return set()

    df = pd.read_csv(OUTPUT_FILE)
    if "url" not in df.columns:
        return set()

    return set(df["url"].dropna().tolist())


def append_row_to_csv(row_dict):
    df = pd.DataFrame([row_dict])

    # If file doesn't exist → write header
    write_header = not os.path.exists(OUTPUT_FILE)

    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=write_header,
        index=False
    )


# --------------------------
# Listing Page
# --------------------------

def extract_listing_links():
    soup = get_soup(TAG_URL)
    links = []

    for h1 in soup.select("h1.listing"):
        a = h1.select_one("a.post-title-link")
        if a:
            links.append(BASE_URL + a["href"])

    return list(set(links))


# --------------------------
# Post Processing
# --------------------------

def clean_content(content_div):
    for tag in content_div(["script", "style", "nav"]):
        tag.decompose()

    toc = content_div.select_one("nav.contents")
    if toc:
        toc.decompose()

    paragraphs = content_div.find_all(["p", "h1", "h2", "h3", "li"])

    blocks = []
    for el in paragraphs:
        text = el.get_text(strip=True)
        if text:
            blocks.append(text)

    return "\n\n".join(blocks)


def extract_post_data(url):
    soup = get_soup(url)

    title = soup.select_one("h1.post-title")
    title = title.text.strip() if title else None

    author = soup.select_one("a.author")
    author = author.text.strip() if author else None

    date_tag = soup.select_one("span.date")
    timestamp = None
    readable_date = None

    if date_tag and date_tag.has_attr("data-js-date"):
        timestamp = int(date_tag["data-js-date"])
        readable_date = datetime.utcfromtimestamp(timestamp / 1000)

    karma = soup.select_one("span.karma-value")
    points = karma.get_text(strip=True).split()[0] if karma else None

    comments_tag = soup.select_one("a.comment-count")
    comments = comments_tag.get_text(strip=True).split()[0] if comments_tag else None

    read_time_tag = soup.select_one("span.read-time")
    read_time = read_time_tag.get_text(strip=True) if read_time_tag else None

    tags = [tag.text.strip() for tag in soup.select("#tags a")]

    content_div = soup.select_one("div.body-text.post-body")
    content = clean_content(content_div) if content_div else None

    return {
        "title": title,
        "author": author,
        "date": readable_date,
        "timestamp": timestamp,
        "points": points,
        "comments": comments,
        "read_time": read_time,
        "tags": ", ".join(tags),
        "url": url,
        "content": content
    }


# --------------------------
# Main
# --------------------------

def main():
    print("Loading existing scraped URLs...")
    scraped_urls = load_existing_urls()
    print(f"Already scraped: {len(scraped_urls)}")

    print("Extracting listing links...")
    links = extract_listing_links()
    print(f"Found {len(links)} total links")

    for i, link in enumerate(links):
        if link in scraped_urls:
            print(f"[{i+1}/{len(links)}] Skipping (already scraped)")
            continue

        print(f"[{i+1}/{len(links)}] Scraping {link}")

        try:
            post_data = extract_post_data(link)
            append_row_to_csv(post_data)

            scraped_urls.add(link)  # update in memory
            print("Saved.")

            time.sleep(1)  # polite delay

        except Exception as e:
            print("Error:", e)
            time.sleep(3)  # small cooldown


    print("Done.")


if __name__ == "__main__":
    main()