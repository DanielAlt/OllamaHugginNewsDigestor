import re
import requests 
import threading
import datetime

from urllib.parse import urlparse, unquote

from bs4 import BeautifulSoup

from .core import get_db_connection

def extract_article_content(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup([
        'script',
        'style',
        'noscript',
        'svg',
        'iframe',
        'header',
        'footer',
        'nav',
        'aside'
    ]):
        tag.decompose()

    # Extract visible text
    text = soup.get_text(separator='\n')

    # Normalize whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    text = text.strip()
    return text

class ThreadManager():
    def __init__(self, max_concurrent_threads: int):
        self.semaphore = threading.Semaphore(max_concurrent_threads)
        self.threads: list[ArticleLookupTool] = []

    def add_task(self, url: str, title: str, timeout: int):

        thread = ArticleLookupTool(
            url=url, 
            title=title,
            timeout=timeout, 
            semaphore=self.semaphore
        )
        self.threads.append(thread)

    def run_all(self):
        # Start all threads
        for thread in self.threads:
            thread.start()

        # Wait for all threads to complete
        for thread in self.threads:
            thread.join()

        # Reset internal state if reused
        self.threads.clear()

class ArticleLookupTool(threading.Thread):
    """
        Tool designed to read articles
    """
    def __init__(self, url: str, title: str, timeout: int, semaphore: threading.Semaphore):
        super().__init__()
        self.url        = url
        self.title      = title 
        self.timeout    = timeout
        self.semaphore  = semaphore

    def run(self):
        with self.semaphore: 
            db_con     = get_db_connection()
            db_cur     = db_con.cursor()
            response = requests.get(self.url, timeout=self.timeout, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0"
            })
            response_html = response.text
            response_txt  = extract_article_content(response_html)

            db_cur.execute("""
                INSERT INTO articles(provider, title, url, content, fetched_at) 
                VALUES (:provider, :title, :url, :content, :fetched_at);
            """, {
                "provider": urlparse(self.url).netloc,
                "title": self.title,
                "url": self.url,
                "content": response_txt,
                "fetched_at": datetime.datetime.now()
            })
            db_con.commit()