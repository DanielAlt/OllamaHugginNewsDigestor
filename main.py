"""
ollama-huggin-bridge 

# QuickStart

```bash
ollama run qwen3:4b
winget install "FFmpeg (Essentials Build)" 
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
5. Each article summary should be written to cache
6. An executive report should be generated and posted in a discord channel
7. A 'Read aloud' MP3 file should be generated with the contents of 
 the executive summary.

## Bugs

- Some articles won't load via the 'requests' module, sites using ReactJS or 
 other such frameworks load all content from asynchronous calls after the initial
 page load. For such sites we need to 'detect' and load them in a headless 
 browser. see: https://github.com/browserless/browserless 
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
import time
import os
import numpy as np
import nltk  
import subprocess

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE
from scipy.io.wavfile import write as write_wav

from dotenv import load_dotenv
from pydantic import BaseModel
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urlparse, unquote
from ollama import chat
from ollama import ChatResponse
from contextlib import ExitStack

APP_NAME="OllamaHugginBridge"
APP_VERSION="0.0.1"
APP_CACHE_DIR="cache"
APP_DEBUG_OUTPUT=False
APP_DESCRIPTION="""\
An AI (Ollama) digest of a Discord Message Channel devoted to collecting huggin\
 news articles. 
"""

DISCORD_READ_CHANNEL_ID=1397390767009300531
DISCORD_WRITE_CHANNEL_ID=1502884461832835112
DISCORD_BOT_TOKEN=""
DISCORD_EPOCH = 1420070400000

OLLAMA_MAX_CTX=16384
OLLAMA_RESERVED_OUTPUT=1024
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

Criticality is defined by the impact of not addressing a vulnerability \
 multiplied by it's likelihood of being exploited. Positive stories, and \
 advisories about software updates (unless specifically addressing a \
 vulnerability) should be treated as non-critical. 

 The executive summary should be 
 - 600 words long, at maximum.
 - formulated as a 'speech' that would be read by a News Caster. This means \
 short paragraphs, in active voice. Don't use any markdown formatting, \
 bold or italics, or html tags. 
 - Always begin the summary output with "Today in CyberSecurity,"
 - Never contain URLs or SHA256 hash data, since it is unnatural to speak those\
 aloud. 

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

    parser.add_argument(
        "--debug-output",
        type=bool,
        required=False,
        default=False,
        help="Send debug ouput to the CLI"
    )

    args = parser.parse_args()

    return vars(args)

def debug_output(message, error=False):
    if not APP_DEBUG_OUTPUT: 
        return 
    if not error:
        sys.stdout.write(f"{message}\n")
    else:
        sys.stderr.write(f"{message}\n")

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

    def send_message(self, content: str, retry_count: int = 0) -> dict:

        MAX_RETRIES = 5

        lines = content.split("\n")

        chunk_groups = []
        current_group = []

        for line in lines:

            test_group = current_group + [line]
            test_content = "\n".join(test_group)

            if len(test_content) > 1999:

                if current_group:
                    chunk_groups.append(current_group)

                current_group = [line]

            else:
                current_group.append(line)

        if current_group:
            chunk_groups.append(current_group)

        response = None

        for i, group in enumerate(chunk_groups):

            payload_content = "\n".join(group)
            response = requests.post(
                f"https://discord.com/api/v10/channels/{DISCORD_WRITE_CHANNEL_ID}/messages",
                headers=self.headers,
                json={
                    "content": payload_content
                }
            )

            if response.status_code == 422:

                if retry_count >= MAX_RETRIES:
                    raise Exception(
                        f"Maximum retries exceeded for Discord message send: {response.text}"
                    )

                backoff_seconds = 2 ** retry_count
                time.sleep(backoff_seconds)

                return self.send_message(
                    content=content,
                    retry_count=retry_count + 1
                )

            response.raise_for_status()

        return response.json()

    def send_attachments(self, attachments):
        
        with ExitStack() as stack:
            files = []
            
            for i, filename in enumerate(attachments):
                f = stack.enter_context(open(filename, "rb"))
                files.append((f"files[{i}]", f))

            response = requests.post(
                f"https://discord.com/api/v10/channels/{DISCORD_WRITE_CHANNEL_ID}/messages",
                data={"content": "Your files, as requested"}, 
                files=files, 
                headers=self.headers
            )

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
        debug_output(f"Summarizing Article: {article_title}")
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
        debug_output(f"Ran for {delta} seconds")

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
    debug_output(f"Ran for {delta} seconds")
    return response.message.content

def get_meta_from_article_summaries(session_cache_dir_summaries: Path) -> list:
    reference_list = []
    ioc_list = []
    for article_summary_path in session_cache_dir_summaries.iterdir():
        with open(article_summary_path, 'r', encoding="utf8") as fhandle:
            summary_json = json.loads(fhandle.read())
            reference_list.append(summary_json['url'])
            ioc_list += summary_json['iocs']
            fhandle.close()
    return {'ref': reference_list, 'ioc': ioc_list }

def tts(text_prompt, output_filename):
    text_prompt = text_prompt.replace("\n", " ").strip()

    sentences = nltk.sent_tokenize(text_prompt)
    speaker="v2/en_speaker_9"
    silence = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)

    pieces = []
    for i, sentence in enumerate(sentences):
        audio_array = np.squeeze(generate_audio(sentence, history_prompt=speaker))
        pieces.append(audio_array)

        if i != len(sentences) - 1:
            pieces.append(silence)

    full_audio = np.concatenate(pieces, axis=0)
    write_wav(f"{output_filename}.wav", SAMPLE_RATE, full_audio)
    subprocess.run([ "ffmpeg", "-i", "input.wav", "-b:a", "64k", f"{output_filename}.mp3"], check=True)

def main(config):
    global APP_NAME
    global APP_DEBUG_OUTPUT
    global APP_VERSION
    global APP_CACHE_DIR
    global DISCORD_READ_CHANNEL_ID
    global DISCORD_WRITE_CHANNEL_ID
    global DISCORD_BOT_TOKEN
    global OLLAMA_MAX_CTX
    global OLLAMA_RESERVED_OUTPUT
    global SAFE_INPUT_TOKENS
    
    APP_NAME                    = str(os.getenv('APP_NAME'))
    APP_DEBUG_OUTPUT            = bool(os.getenv('APP_DEBUG_OUTPUT'))
    APP_VERSION                 = str(os.getenv('APP_VERSION'))
    APP_CACHE_DIR               = str(os.getenv('APP_CACHE_DIR'))
    DISCORD_READ_CHANNEL_ID     = str(os.getenv('DISCORD_READ_CHANNEL_ID'))
    DISCORD_WRITE_CHANNEL_ID    = str(os.getenv('DISCORD_WRITE_CHANNEL_ID'))
    DISCORD_BOT_TOKEN           = str(os.getenv('DISCORD_BOT_TOKEN'))
    OLLAMA_MAX_CTX              = int(os.getenv('OLLAMA_MAX_CTX'))
    OLLAMA_RESERVED_OUTPUT      = int(os.getenv('OLLAMA_RESERVED_OUTPUT'))
    SAFE_INPUT_TOKENS = OLLAMA_MAX_CTX - OLLAMA_RESERVED_OUTPUT

    cache_dir = setup_cache_dir() # installation step
    debug_output(f"Starting {APP_NAME} with configuration")
    debug_output(str(config))

    # 1. Go to Discord, read the given channel
    end_time            = datetime.datetime.now(datetime.UTC)
    start_time          = end_time - datetime.timedelta(days=config["days"])
    after_snowflake     = datetime_to_snowflake(start_time)
    before_snowflake    = datetime_to_snowflake(end_time)

    debug_output("Fetching discord messages...")
    discord_client = DiscordAPIClient()
    messages = discord_client.read_messages({
        "after": after_snowflake,
        "before": before_snowflake
    })
    debug_output(f"Retreived {len(messages)} messages from discord")

    article_links = []
    for message in messages: 
        if message['author']['username'] != "Threat Intelligence Bot":
            continue
        if message['embeds']:
            article_links.append({
                "url": message['embeds'][0]['url'],
                "title": message['embeds'][0]['title']
            })
    debug_output(f"Filtered to {len(article_links)} articles")

    debug_output("Setting up cache directories for session")
    # 2. For all articles read since our last run, visit the article link 
    # 2.1 Create a working cache directory where to save articles and summaries
    session_cache_dir_articles = cache_dir / str(end_time.strftime("%Y%m%d%H%M%S")) / "articles"
    session_cache_dir_summaries = cache_dir / str(end_time.strftime("%Y%m%d%H%M%S")) / "summaries"
    Path(session_cache_dir_articles).mkdir(exist_ok=True, parents=True)
    Path(session_cache_dir_summaries).mkdir(exist_ok=True, parents=True)

    # 2.2 Create the thread manager
    debug_output("Preparing to Lookup Articles, spawning thread manager")
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
    debug_output("Running multi-threaded article lookups")
    thread_manager.run_all()

    # 4. At this point articles are all stored as 'txt' files, 
    # in the session_cache_dir path. summarize them with Ollama
    # We do this 1 at a time to avoid melting the computer...
    debug_output("Preparing to summarize articles with Ollama")
    summarize_articles(
        config, 
        session_cache_dir_articles, 
        session_cache_dir_summaries
    )

    debug_output(f"Generating Final Executive Summary")
    exec_summary = executive_summary(config, session_cache_dir_summaries)

    exec_summary_filename = f'exec-summary{str(end_time.strftime("%Y%m%d%H%M%S"))}' 
    exec_summary_final = exec_summary[exec_summary.find("</think>")+8:]
    exec_summary_think = exec_summary[exec_summary.find("<think>")+7:exec_summary.find("</think>")]

    with open(f'{exec_summary_filename}.think.txt', 'w', encoding="utf8") as fhandle:
        fhandle.write(exec_summary_think)
        fhandle.close()

    tts(exec_summary_final, exec_summary_filename)

    article_meta = get_meta_from_article_summaries(session_cache_dir_summaries)
    exec_summary_final += "\n\nReferences:\n"
    for ref in article_meta['ref']:
        exec_summary_final += f"{ref}\n"

    exec_summary_final += "\n\nIOCs:\n"
    for ioc in article_meta['ioc']:
        exec_summary_final += f"{ioc}\n"

    with open(f'{exec_summary_filename}.txt', 'w', encoding="utf8") as fhandle:
        fhandle.write(exec_summary_final)
        fhandle.close()

    debug_output(f"Sending Executive Summary to Discord Channel")
    discord_client.send_attachments([
        f"{exec_summary_filename}.txt", f"{exec_summary_filename}.mp3"
    ])
    # discord_client.send_message(exec_summary_final)

if __name__ == "__main__":
    config = parse_arguments()
    load_dotenv()
    preload_models()
    main(config)
    sys.exit(0)