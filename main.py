"""
ollama-huggin-bridge 

# QuickStart

```bash
ollama run qwen3:4b
python -m pip install -r requirements.txt
python main.py --days 1
```

# Developer Notes

## Objective

In this program We want to achieve the following:

1. Go to Discord, read the given channel (of RSS feeds)
2. For all articles read since our last run, visit the article link 
3. Retrieve the full article content 
4. Summarize each article with AI; pay specific attention to:
  - Named people/organizations
  - Named Malwares 
  - Indicators of Compromise (IOC)
  - Severity of issue or finding
5. Each article summary should be written cache
6. An executive report should be generated and posted in a discord channel
7. A 'Read aloud' MP3 file should be generated with the contents of 
 the executive summary.

## Bugs

- Some articles won't load via the 'requests' module, sites using ReactJS or 
 other such frameworks load all content from asynchronous calls after the initial
 page load. For such sites we need to 'detect' and load them in a headless 
 browser. 

- hardcoded Discord API credentials
"""
import sys
import argparse
import requests
import json
import datetime
import threading
import platform
import re
import tiktoken

from pydantic import BaseModel
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urlparse, unquote
from ollama import chat
from ollama import ChatResponse

APP_NAME="OllamaHugginBridge"
APP_VERSION="0.0.1"
APP_CACHE_DIR="cache"
APP_DESCRIPTION="""\
An AI (Ollama) digest of a Discord Message Channel devoted to collecting huggin\
 news articles. 
"""

DISCORD_READ_CHANNEL_ID="1397390767009300531"
DISCORD_WRITE_CHANNEL_ID="1502884461832835112"
DISCORD_BOT_TOKEN="INSERT YOUR API TOKEN HERE "
DISCORD_EPOCH = 1420070400000

OLLAMA_MAX_CTX = 16384
OLLAMA_RESERVED_OUTPUT = 1024
SAFE_INPUT_TOKENS = OLLAMA_MAX_CTX - OLLAMA_RESERVED_OUTPUT
OLLAMA_ARTICLE_SUMMARY_PROMPT="""\
Following this text you will receive an article in txt format.\
 Summarize the article content. The summary should be no more than 500\
 characters. pay specific attention to:
  - Named organizations, and vendors
  - Named Malwares. If none are named, leave this blank. 
  - Indicators of Compromise (IOC): this can be an IP address, Domain name,\
 sha256 checksum, or filename. If none of those values are available leave this\
  blank
  - Severity of issue or finding

ARTICLE CONTENT:
"""

OLLAMA_EXECUTIVE_SUMMARY_PROMPT="""\
Following this text you will receive a list of article summaries in txt format.\
 Each article summary is separated by two new line characters (\\n). Draft an\
 executive summary of all the articles, grouping them by theme, and highlighting\
 the most critical. Skip any summaries that appear to be junk, make no mention\
 of them. 

 The executive summary should be 
 - 600 words long, at maximum.
 - contain no formatting, just text.  
 - formulated as a 'speech' that would be read by a News Caster. 
 - Always begin with "Today in CyberSecurity,"

 ARTICLE SUMMARIES:
"""

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} {APP_VERSION} - {APP_DESCRIPTION}"
    )

    parser.add_argument(
        "--days",
        type=int,
        required=True,
        help="Amount of days prior to today to include in the process."
    )

    parser.add_argument(
        "--max-threads",
        type=int,
        required=False,
        default=4,
        help="Maximum number of threads that can run concurrently when reading articles"
    )

    parser.add_argument(
        "--thread-timeout",
        type=int,
        required=False,
        default=30,
        help="Maximum timeout value for thread"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="qwen3:4b",
        help="The name of the Ollama Model, default is qwen3.6"
    )

    args = parser.parse_args()

    return vars(args)

def setup_cache_dir() -> Path:
    if platform.system() == "Windows":
        base = Path.home() / "AppData" / "Local" / APP_NAME
    else:
        base = Path.home() / f".{APP_NAME}"

    cache_dir = base / APP_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir

def datetime_to_snowflake(dt: datetime.datetime) -> int:
    unix_ms = int(dt.timestamp() * 1000)
    return (unix_ms - DISCORD_EPOCH) << 22

def url_to_filename(url: str, max_length: int = 150) -> str:
    parsed = urlparse(url)
    raw_name = f"{parsed.netloc}{parsed.path}"
    raw_name = unquote(raw_name)
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', raw_name)
    safe_name = re.sub(r'_+', '_', safe_name)
    safe_name = safe_name.strip("._")
    if not safe_name:
        safe_name = "default_filename"

    return safe_name[:max_length]

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

def truncate_to_token_limit(encoding, text, max_tokens):
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    truncated = encoding.decode(tokens[:max_tokens])

    return truncated

class DiscordAPIClient():
    """ 
        Discord API Client Class, for Reading and Writing Messages to 
        the NewsFeed channel
    """
    def __init__(self):
        self.headers = { "Authorization": f"Bot {DISCORD_BOT_TOKEN}" }

    def read_messages(self, params: dict) -> dict:
        api_url = f"https://discord.com/api/v10/channels/{DISCORD_READ_CHANNEL_ID}/messages"
        """ Read Messages from the hardcoded channel """
        response = requests.get(
            api_url, 
            headers=self.headers, 
            params=params
        )
        messages = response.json()
        return messages

    def send_message(
        self,
        content: str
    ) -> dict:

        # We will split the content size into paragraphs and assure that the sum 
        # of the paragraphs does not exceed 2000, sending the maximum allowable 
        # number of paragraphs not exceeding the 2000 character limit at a time 
        content_chunks = content.split("\n")
        chunk_groups = [[]]
        chunk_size = 0
        for i in range(0, len(content_chunks)):
            current_chunk = content_chunks[i]
            current_chunk_size = chunk_size + len(current_chunk)
            if current_chunk_size > 1999:
                chunk_groups.append([current_chunk])
                chunk_size = len(current_chunk)
            else:
                chunk_group_index = len(chunk_groups)-1
                chunk_groups[chunk_group_index].append(current_chunk)
                chunk_size += len(current_chunk) 

        for chunk_group in chunk_groups:
            """ Send a message to a Discord channel """
            api_url = f"https://discord.com/api/v10/channels/{DISCORD_WRITE_CHANNEL_ID}/messages"
            payload = { "content": "\n".join(chunk_group) }
            response = requests.post(
                api_url,
                headers=self.headers,
                json=payload
            )

        return response.json()

class ThreadManager():
    def __init__(self, max_concurrent_threads: int):
        self.semaphore = threading.Semaphore(max_concurrent_threads)
        self.threads: list[ArticleLookupTool] = []

    def add_task(self, url: str, title: str, timeout: int, cache_dir: Path):

        thread = ArticleLookupTool(
            url=url, 
            title=title,
            timeout=timeout, 
            cache_dir=cache_dir, 
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
    def __init__(self, url: str, title: str, timeout: int, cache_dir: Path, semaphore: threading.Semaphore):
        super().__init__()
        self.url        = url
        self.title      = title 
        self.filename   = cache_dir / f"{url_to_filename(url)}.txt"
        self.timeout    = timeout
        self.semaphore  = semaphore

    def run(self):
        with self.semaphore: 
            response = requests.get(self.url, timeout=self.timeout, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0"
            })
            response_html = response.text
            response_txt  = extract_article_content(response_html)
            with open(self.filename, 'w', encoding="utf8") as fhandle:
                fhandle.write(f"Article URL:{self.url}\n")
                fhandle.write(f"Article Title:{self.title}\n\n")
                fhandle.write(response_txt)
                fhandle.close()

class Article(BaseModel):
    organizations: list[str]
    vendors: list[str]
    iocs: list[str]
    malwares: list[str]
    severity: str
    summary: str

def summarize_articles(config: dict, session_cache_dir_articles: Path, session_cache_dir_summaries: Path):
    """4. Summarize each article with AI; pay specific attention to:
      - Named people/organizations
      - Named Malwares 
      - Indicators of Compromise (IOC)
      - Severity of issue or finding
    To assure the prompt doesn't exceed the context window, we truncate if it 
    does. 
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    for article_content_path in session_cache_dir_articles.iterdir():
        article_content_raw = ""
        with open(article_content_path, 'r', encoding="utf8") as fhandle:
            article_content_raw = fhandle.read()
            fhandle.close()

        article_url   = article_content_raw.split("\n")[0][12:]
        article_title = article_content_raw.split("\n")[1][14:]
        
        article_content = "\n".join(article_content_raw.split("\n")[2:])

        prompt = f"{OLLAMA_ARTICLE_SUMMARY_PROMPT}\n\n{article_content}"
        prompt_tokens = len(encoding.encode(prompt))

        if prompt_tokens > SAFE_INPUT_TOKENS:
            article_content = truncate_to_token_limit(
                article_content,
                SAFE_INPUT_TOKENS // 2
            )
        print(f"Summarizing Article: {article_title}")
        prompt = f"{OLLAMA_ARTICLE_SUMMARY_PROMPT}\n\n{article_content}"
        now = datetime.datetime.now()
        response: ChatResponse = chat(
            model=config['model_name'], 
            messages=[{
            'role': 'user', 
            'content': prompt,
            }],
            format=Article.model_json_schema(),
            think=False,
            options={
                "num_ctx": 16384,
                "temperature": 0
            }
        )
        then = datetime.datetime.now()
        delta = then - now 
        print(f"Ran for {delta} seconds")

        article = Article.model_validate_json(response.message.content)
        article_dict = article.model_dump()
        article_dict['url'] = article_url
        article_dict['title'] = article_title

        # 5. Each article summary should be written to cache
        summary_filename = session_cache_dir_summaries / f"summary-{article_content_path.parts[-1][:-4]}.json"
        with open(summary_filename, 'w', encoding="utf8") as fhandle:
            json.dump(article_dict, fhandle, indent=2)
            fhandle.close()

def executive_summary(config: dict, session_cache_dir_summaries: Path) -> str:
    # 6. An executive report should be generated and posted in a discord channel
    encoding = tiktoken.get_encoding("cl100k_base")

    article_summaries = ""
    for article_summary_path in session_cache_dir_summaries.iterdir():
        with open(article_summary_path, 'r', encoding="utf8") as fhandle:
            summary_json = json.loads(fhandle.read())
            article_summary_content = summary_json['summary']
            article_summaries  += f"{article_summary_content}\n\n"
            fhandle.close()

    prompt = f"{OLLAMA_EXECUTIVE_SUMMARY_PROMPT}\n\n{article_summaries}"
    prompt_tokens = len(encoding.encode(prompt))

    if prompt_tokens > SAFE_INPUT_TOKENS:
        article_summaries = truncate_to_token_limit(
            article_summaries,
            SAFE_INPUT_TOKENS // 2
        )

    prompt = f"{OLLAMA_EXECUTIVE_SUMMARY_PROMPT}\n\n{article_summaries}"
    now = datetime.datetime.now()
    response: ChatResponse = chat(
        model=config['model_name'], 
        messages=[{
            'role': 'user', 
            'content': prompt,
        }],
        think=False,
        options={
            "num_ctx": 16384,
            "temperature": 0
        }
    )
    then = datetime.datetime.now()
    delta = then - now 
    content = re.sub(
        r"<think>.*?</think>",
        "",
        response.message.content,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()

    print(f"Ran for {delta} seconds")
    return content

def main(config):
    cache_dir = setup_cache_dir() # installation step

    # 1. Go to Discord, read the given channel
    end_time            = datetime.datetime.now(datetime.UTC)
    start_time          = end_time - datetime.timedelta(days=config["days"])
    after_snowflake     = datetime_to_snowflake(start_time)
    before_snowflake    = datetime_to_snowflake(end_time)

    discord_client = DiscordAPIClient()
    messages = discord_client.read_messages({
        "after": after_snowflake,
        "before": before_snowflake
    })

    article_links = []
    for message in messages: 
        if message['author']['username'] != "Threat Intelligence Bot":
            continue
        if message['embeds']:
            article_links.append({
                "url": message['embeds'][0]['url'],
                "title": message['embeds'][0]['title']
            })

    # 2. For all articles read since our last run, visit the article link 
    # 2.1 Create a working cache directory where to save articles and summaries
    session_cache_dir_articles = cache_dir / str(end_time.strftime("%Y%m%d%H%M%S")) / "articles"
    session_cache_dir_summaries = cache_dir / str(end_time.strftime("%Y%m%d%H%M%S")) / "summaries"
    Path(session_cache_dir_articles).mkdir(exist_ok=True, parents=True)
    Path(session_cache_dir_summaries).mkdir(exist_ok=True, parents=True)

    # 2.2 Create the thread manager
    max_threads = config['max_threads'] if config['max_threads'] is not None else 4 
    thread_manager = ThreadManager(max_concurrent_threads=max_threads)
    for article_dict in article_links:
        # Add threads to the thread Manager
        thread_manager.add_task(
            url=article_dict['url'],
            title=article_dict['title'],
            timeout=config['thread_timeout'],
            cache_dir=session_cache_dir_articles
        )
    # 3. Retrieve the full article content 
    thread_manager.run_all()

    # 4. At this point articles are all stored as 'txt' files, 
    # in the session_cache_dir path. summarize them with Ollama
    # We do this 1 at a time to avoid melting the computer...
    summarize_articles(
        config, 
        session_cache_dir_articles, 
        session_cache_dir_summaries
    )

    exec_summary = executive_summary(config, session_cache_dir_summaries)
    discord_client.send_message(exec_summary)

if __name__ == "__main__":
    config = parse_arguments()
    main(config)
    sys.exit(0)