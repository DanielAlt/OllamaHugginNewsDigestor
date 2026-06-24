"""
ollama-huggin-bridge 

# QuickStart

```bash
py install 3.11
py -3.11 -m venv venv
./venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
winget install "FFmpeg (Essentials Build)" 

# Install Ollama
ollama run qwen3:4b
python main.py --days 1
```

# Developer Notes

## Objective

In this program We want to achieve the following:

1. Go to Discord, read the given channel (of RSS feeds)
2. For all articles read since our last run, visit the article link 
3. Retrieve the full article content via http
4. Summarize each article with AI; extract actionable threat intelligence
5. Each article summary should be written to cache
6. An executive report should be generated and posted in a discord channel
7. A 'Read aloud' MP3 file should be generated with the contents of 
 the executive summary.
"""
import sys
import argparse
import requests
import json
import datetime
import threading
import platform
import re
import time
import os
import subprocess
import shutil
import sqlite3

import tiktoken
import torch

from pathlib import Path
from urllib.parse import urlparse, unquote
from contextlib import ExitStack

from TTS.api import TTS
from dotenv import load_dotenv
from pydantic import BaseModel, Field, constr
from typing import Literal
from bs4 import BeautifulSoup
from ollama import chat, ChatResponse

APP_NAME="OllamaHugginBridge"
APP_VERSION="0.0.1"
APP_DEBUG_OUTPUT=False
APP_DESCRIPTION="""\
An AI (Ollama) digest of a Discord Message Channel devoted to collecting huggin\
 news articles. See the readme.md for more information. 
"""

DISCORD_READ_CHANNEL_ID=1397390767009300531
DISCORD_WRITE_CHANNEL_ID=1502884461832835112
DISCORD_BOT_TOKEN=""
DISCORD_EPOCH = 1420070400000

OLLAMA_MAX_CTX=16384
OLLAMA_RESERVED_OUTPUT=1024
SAFE_INPUT_TOKENS = OLLAMA_MAX_CTX - OLLAMA_RESERVED_OUTPUT

OLLAMA_ARTICLE_SUMMARY_PROMPT="""
You are a Cyber Security Researcher. Your job is to analyze threat intelligence
articles, security news, and blog posts in order to extract the most relevant
details for a Security Operations Center (SOC).

## General Rules

* Extract only information explicitly stated in the article.
* Do not infer threat actors, malware, vulnerabilities, victims, or IOCs.
* Do not use knowledge outside the article contents.
* If uncertain, omit the value.
* Prefer precision over recall.
* Deduplicate all extracted values.
* If no values are found for a list field, return an empty list.
* Never return placeholder values such as "Unknown", "N/A", or
  "Not Mentioned".
* Ignore navigation menus, tags, related articles, advertisements,
  and site metadata.

## Summary

summary:

* Maximum 500 characters.
* Use a concise, factual, and impartial tone.
* Avoid unsupported conclusions, industry trends, predictions,
  or speculation.

## Extract Relevant Data

vendor_organization:

* Names of software vendors, cybersecurity vendors, and organizations
  mentioned in the article.

threat_actor_list:

* Named threat actors, APT groups, intrusion sets, and criminal groups.

ioc_list:

* Only include IOCs explicitly present in the article text.
* Do NOT generate placeholder IOC entries.
* Do NOT output empty values.
* Do NOT output IOC types unless a real value is present.
* If no IOC exists for a type, omit it entirely.
* Each IOC object must contain a real extracted value.
* Assign confidence:

  * high: explicitly attributed to malicious activity
  * medium: likely malicious but attribution is indirect
  * low: mentioned with uncertainty or weak evidence

ttp_list:

* Short description of attacker techniques, procedures, or tools.
* Maximum 100 characters per entry.

malware_list:

* Named malware families, implants, loaders, trojans, ransomware,
  and backdoors.
* Do not include security tools, administration tools, penetration
  testing tools, or frameworks unless the article explicitly identifies
  them as malware.

vulnerability_list:

* Only include vulnerabilities with a valid CVE identifier.
* Ignore vulnerability names without an associated CVE.

severity:

* Assess severity using only information contained in the article.
* critical: active widespread exploitation, ransomware campaigns,
  nation-state activity, or critical infrastructure impact
* high: confirmed exploitation or significant organizational risk
* medium: credible threat activity without widespread exploitation
* low: informational, historical, research, or low-impact content
"""

OLLAMA_EXECUTIVE_SUMMARY_PROMPT="""\
You are a senior cybersecurity news editor preparing the Daily Update script \
 for a professional news broadcast.

You will receive multiple cybersecurity and technology news article summaries.

Each article summary is separated by two newline characters (\\n\\n).

Your task is to synthesize the material into a concise broadcast script that\
 will be read aloud by a news presenter.

## Requirements

* Maximum length: 600 words.
* Write as a teleprompter script intended for spoken delivery.
* Use clear, concise, active voice.
* Use short paragraphs of one to three sentences.
* Maintain a professional newsroom tone.
* Do not use markdown, HTML, bullet points, headings, bold text, or italics.
* Expand all acronyms. For example, "Indicator of Compromise" instead of "IOC".
* Never include URLs, file paths, IP addresses, hash values, or other\ 
 machine-readable artifacts that sound unnatural when spoken aloud.
* Begin the script exactly with:
  Today in CyberSecurity,

## Editorial Guidelines

* Do not make broad claims about industry-wide trends, attacker behavior, or\
 the cybersecurity landscape unless those claims are directly supported by\ 
 the provided articles.
* Do not infer conclusions that are not explicitly supported by the source\ 
 material.
* Avoid speculation about future attacks, future impacts, or future\ 
 industry developments.
* Avoid generic news-anchor commentary and dramatic language.
* Do not use phrases such as:

  * "the stakes have never been higher"
  * "organizations must remain vigilant"
  * "the threat landscape continues to evolve"
  * "cybersecurity remains a top priority"
  * "only time will tell"
  * "this serves as a reminder"
  * "a wake-up call"
  * "an ever-changing threat landscape"
* Focus on reporting concrete facts, actions, impacts, and outcomes described\ 
 in the articles.
* When summarizing multiple stories, prefer specific details over broad\ 
 conclusions.
* Do not add a moral, lesson, or editorial opinion at the end of the broadcast.


Content Selection

* Merge duplicate stories covering the same event into a single segment.
* Group related stories together naturally.
* Prioritize stories according to their real-world impact.
* If all stories cannot be covered within the word limit, summarize \
 lower-priority stories briefly or omit them.

Story Prioritization

Highest Priority

* Actively exploited vulnerabilities.
* Vulnerabilities listed in Known Exploited Vulnerabilities catalogues.
* Vulnerabilities for which proof of concept exploit code exists. 
* Remote code execution vulnerabilities with evidence of exploitation.
* Major ransomware campaigns.
* Large-scale data breaches.
* Significant nation-state activity.
* Incidents affecting critical infrastructure.

Medium Priority

* High-severity vulnerabilities without confirmed exploitation.
* Emerging malware campaigns.
* Significant defensive innovations.
* Important vendor security advisories.

Lower Priority

* Product announcements.
* Minor software updates.
* Industry commentary.
* Research findings without immediate operational impact.

Output Structure

1. Open with the most critical cybersecurity developments.
2. Continue with major threat activity, breaches, and vulnerability news.
3. Cover defensive measures, research, and industry developments.
4. End with a brief closing sentence summarizing the day's security landscape.

The final output must read naturally as a single news broadcast and must never\
 mention article summaries, source material, researchers, or editorial\ 
 decisions.
"""

DOCUMENT_DISCLAIMER="""\
**This document is Generated by Artificial Intelligence (AI). Accuracy is not \
 guaranteed.**\n\n
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

    args = parser.parse_args()
    return vars(args)

def debug_output(message, error=False):
    if not APP_DEBUG_OUTPUT: 
        return 
    if not error:
        sys.stdout.write(f"{message}\n")
    else:
        sys.stderr.write(f"{message}\n")

def get_installation_path() -> Path:
    if platform.system() == "Windows":
        base = Path.home() / "AppData" / "Local" / APP_NAME
    else:
        base = Path.home() / f".{APP_NAME}"
    return base

def setup_self() -> Path:
    base = get_installation_path()
    base.mkdir(parents=True, exist_ok=True)
    return base

def setup_cache_dir() -> Path:
    debug_output("Creating Cache Directory")
    cache_dir = get_installation_path() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def setup_dotenv():
    env = get_installation_path() / ".env"
    if env.is_file():
        load_dotenv(dotenv_path=env)
    else:
        print(f"This is your first time using {APP_NAME}")
        print(f"Please Update the file at {env} to configure the application")
        env_example = ""
        with open(".env.example", 'r', encoding="utf8") as fhandle1:
            env_example = fhandle1.read()
            fhandle1.close()
        with open(env, 'w', encoding='utf8') as fhandle2:
            fhandle2.write(env_example)
            fhandle2.close()
        return sys.exit(0)

def setup_database():
    schema = """
        CREATE TABLE IF NOT EXISTS articles(
            id INTEGER PRIMARY KEY,
            provider TEXT NOT NULL,
            title VARCHAR(255) DEFAULT NULL,
            url TEXT NOT NULL UNIQUE,
            content TEXT DEFAULT NULL,
            fetched_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS article_summaries(
            id INTEGER PRIMARY KEY, 
            article_id INTEGER NOT NULL UNIQUE,
            content TEXT DEFAULT NULL, 
            reasoning TEXT DEFAULT NULL, 
            json TEXT DEFAULT NULL,
            FOREIGN KEY (article_id)
                REFERENCES articles(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_articles_fetched_at
            ON articles(fetched_at);

        CREATE TABLE IF NOT EXISTS iocs(
            id INTEGER PRIMARY KEY,
            article_id INTEGER NOT NULL,
            value TEXT NOT NULL,
            confidence TEXT NOT NULL,
            type TEXT NOT NULL,
            added_on DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (article_id)
                REFERENCES articles(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_iocs_article_id
            ON iocs(article_id);

        CREATE INDEX IF NOT EXISTS idx_iocs_value
            ON iocs(value);

        CREATE INDEX IF NOT EXISTS idx_iocs_type
            ON iocs(type);
    """

    db_path = get_installation_path() / "database.sqlite"
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.executescript(schema)
    connection.commit()
    return connection

def get_db_connection():
    db_path = get_installation_path() / "database.sqlite"
    connection = sqlite3.connect(db_path)
    return connection

def setup_system_prompts():
    global OLLAMA_ARTICLE_SUMMARY_PROMPT
    global OLLAMA_EXECUTIVE_SUMMARY_PROMPT

    base_path = get_installation_path()
    summary_prompt = base_path / "article-summary-prompt.md"
    executive_prompt = base_path / "executive-summary-prompt.md"

    if summary_prompt.is_file():
        with open(summary_prompt, 'r', encoding="utf8") as fhandle1:
            OLLAMA_ARTICLE_SUMMARY_PROMPT = fhandle1.read()
            fhandle1.close()
    else:
        with open(summary_prompt, 'w', encoding="utf8") as fhandle1:
            fhandle1.write(OLLAMA_ARTICLE_SUMMARY_PROMPT)
            fhandle1.close()
    
    if executive_prompt.is_file():
        with open(executive_prompt, 'r', encoding="utf8") as fhandle2:
            OLLAMA_EXECUTIVE_SUMMARY_PROMPT = fhandle2.read()
            fhandle2.close()
    else:
        with open(executive_prompt, 'w', encoding="utf8") as fhandle2:
            fhandle2.write(OLLAMA_EXECUTIVE_SUMMARY_PROMPT)
            fhandle2.close()


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

def remove_cache_after_this_date(caches_path: Path, day: datetime.datetime):
    debug_output(f"Removing cache content older than {day:%Y-%m-%d %H:%M:%S}")
    directories_cleaned = 0
    for child in caches_path.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue

        try:
            cache_date = datetime.datetime.strptime(child.stem, "%Y%m%d%H%M%S")
            cache_date = cache_date.replace(tzinfo=day.tzinfo)
        except ValueError:
            continue

        if cache_date < day:
            shutil.rmtree(child)
            directories_cleaned += 1

    debug_output(f"Deleted {directories_cleaned} directories")

def deduplicate_article_links(db_con, article_links) -> list:
    db_cur = db_con.cursor()
    deduplicated_article_links = []
    for article_link in article_links:
        sql_to_exec = "SELECT id FROM articles WHERE url = :url;"
        db_cur.execute(sql_to_exec, article_link)
        match_result = db_cur.fetchone()
        if match_result:
            debug_output(f"{article_link['url']} was already reported on")
        else:
            deduplicated_article_links.append(article_link)
    return deduplicated_article_links


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
                data={"content": "Your files, as requested."}, 
                files=files, 
                headers=self.headers
            )

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

class IOC(BaseModel):
    type: Literal[
        "ipv4",
        "ipv6",
        "domain",
        "url",
        "email",
        "sha1",
        "sha256",
        "md5"
    ]
    value: constr(min_length=1)
    confidence: Literal["low", "medium", "high"]

# Todo: Add Vulnerability Lookup Tool. 
# for example: 
# - To cross reference KEV list 
# - To get detailed vulnerability references from CVE Details. 
# - To get CVSS scores when not mentioned. 
# - To determine if Proof Of Concept code is available. 
class Vulnerability(BaseModel):
    software_name: str
    cve_id: str
    cvss_score: float = None

# Todo: Map TTP descriptions to MITRE ATT&CK
# https://attack.mitre.org/versions/v19/
class TTP(BaseModel):
    technique_id: str = None
    technique_name: str = None
    description: str

class Article(BaseModel):
    vendor_organizations: list[str] = Field(default_factory=list)
    threat_actor_list: list[str] = Field(default_factory=list)
    ioc_list: list[IOC] = Field(default_factory=list)
    ttp_list: list[TTP] = Field(default_factory=list)
    malware_list: list[str] = Field(default_factory=list)
    vulnerability_list: list[Vulnerability] = Field(default_factory=list)
    severity: Literal["low", "medium", "high", "critical"]
    summary: str

def summarize_articles(start_time, end_time):
    """4. Summarize each article with AI; pay specific attention to:
      - Named people/organizations
      - Named Malwares 
      - Indicators of Compromise (IOC)
      - Severity of issue or finding
    To assure the prompt doesn't exceed the context window, we truncate if it 
    does. 
    """
    db_con = get_db_connection()
    db_cur = db_con.cursor()
    db_cur.execute("""
        SELECT a.id, a.content, a.title, a.url
        FROM articles a
        WHERE
            a.fetched_at BETWEEN :start_time AND :end_time
            AND NOT EXISTS (
                SELECT 1
                FROM article_summaries s
                WHERE s.article_id = a.id
            );    
    """, {"start_time": start_time, "end_time": end_time})
    article_contents = db_cur.fetchall()

    encoding = tiktoken.get_encoding("cl100k_base")
    for article_content in article_contents:
        article_content_id  = article_content[0]
        article_content_raw = article_content[1]
        article_title       = article_content[2]
        article_url         = article_content[3]

        if len(encoding.encode(OLLAMA_ARTICLE_SUMMARY_PROMPT + article_content_raw)) > SAFE_INPUT_TOKENS:
            article_content_raw = truncate_to_token_limit(
                article_content,
                SAFE_INPUT_TOKENS - len(encoding.encode(OLLAMA_ARTICLE_SUMMARY_PROMPT))
            )
        
        debug_output(f"Summarizing Article: {article_title}")
        now = datetime.datetime.now()
        response: ChatResponse = chat(
            model=os.getenv('OLLAMA_MODEL_NAME'), 
            messages=[
                {'role': 'system', 'content': OLLAMA_ARTICLE_SUMMARY_PROMPT},
                {'role': 'user', 'content': article_content_raw}
            ],
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

        db_cur.execute(""" 
            INSERT INTO article_summaries(
                article_id, 
                content,
                json
            ) VALUES (
                :article_id,
                :content,
                :json
            );
        """, {
            "article_id": article_content_id,
            "content": article_dict['summary'],
            "json": json.dumps(article_dict)
        })
        db_con.commit()

        

def executive_summary(start_time, end_time) -> str:
    # 6. An executive report should be generated and posted in a discord channel
    encoding = tiktoken.get_encoding("cl100k_base")

    db_con = get_db_connection()
    db_cur = db_con.cursor()
    db_cur.execute(""" 
        SELECT s.content, a.title 
        FROM article_summaries s 
        JOIN articles a ON (s.article_id = a.id)
        WHERE a.fetched_at BETWEEN :start_time AND :end_time;
    """, {
        "start_time": start_time,
        "end_time": end_time
    })
    article_summaries_rs = db_cur.fetchall()

    article_summaries = ""
    for article_summary_rs in article_summaries_rs:
        article_summaries += f"# {article_summary_rs[1]}\n\n{article_summary_rs[0]}\n\n"

    if len(encoding.encode(OLLAMA_EXECUTIVE_SUMMARY_PROMPT + article_summaries)) > SAFE_INPUT_TOKENS:
        article_summaries = truncate_to_token_limit(
            article_summaries,
            SAFE_INPUT_TOKENS - len(encoding.encode(OLLAMA_EXECUTIVE_SUMMARY_PROMPT))
        )
        
    now = datetime.datetime.now()
    response: ChatResponse = chat(
        model=os.getenv("OLLAMA_MODEL_NAME"), 
        messages=[
            { 'role': 'system', 'content': OLLAMA_EXECUTIVE_SUMMARY_PROMPT }, 
            { 'role': 'user', 'content': article_summaries}
        ],
        think=False,
        options={
            "num_ctx": 16384,
            "temperature": 0.3
        }
    )

    then = datetime.datetime.now()
    delta = then - now 
    debug_output(f"Ran for {delta} seconds")

    return response.message.content

def get_further_reading_section(start_time, end_time) -> list:
    db_con = get_db_connection()
    db_cur = db_con.cursor()
    db_cur.execute(""" 
        SELECT s.json 
        FROM article_summaries s
        JOIN articles a ON (s.article_id = a.id)
        WHERE a.fetched_at BETWEEN :start_time AND :end_time;
    """, {
        "start_time": start_time,
        "end_time": end_time
    })
    article_summary_rs = db_cur.fetchall()

    further_reading = "\n\n# Further Reading\n"
    for article_summary in article_summary_rs:
        summary_json = json.loads(article_summary[0])

        further_reading += f"## [{summary_json['title']}]({summary_json['url']})\n"
        further_reading += f"*Severity*: {summary_json['severity']}\n\n"
        further_reading += summary_json['summary'] + "\n\n"
        if len(summary_json['ioc_list']):
            further_reading += "\n\n|IOC Value|Type|Confidence\n|---|---|---|\n"
            for ioc in summary_json['ioc_list']:
                further_reading += f"|{ioc['value']}|{ioc['type']}|{ioc['confidence']}|\n"

        if len(summary_json['vulnerability_list']):
            further_reading += "\n\n|Vulnerability|Software|\n|---|---|\n"
            for vuln in summary_json['vulnerability_list']:
                further_reading += f"|{vuln['cve_id']}|{vuln['software_name']}|\n"
 
    return (further_reading)

def tts(text_prompt, output_filename):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    speaker_wav = "voice-samples/voice-sample3.1m.wav"   # your reference voice
    language = "en"
    tts.tts_to_file(
        text=text_prompt,
        speaker_wav=speaker_wav,
        language=language,
        file_path=f"{output_filename}.wav"
    )
    subprocess.run([ "ffmpeg", "-i", f"{output_filename}.wav", "-b:a", "64k", f"{output_filename}.mp3"], check=True)

def main(config):
    global APP_DEBUG_OUTPUT
    global DISCORD_READ_CHANNEL_ID
    global DISCORD_WRITE_CHANNEL_ID
    global DISCORD_BOT_TOKEN
    global OLLAMA_MAX_CTX
    global OLLAMA_RESERVED_OUTPUT
    global SAFE_INPUT_TOKENS

    # Installation steps
    setup_self()
    setup_dotenv()
    setup_system_prompts()
    cache_dir = setup_cache_dir()
    db_con = setup_database()

    APP_DEBUG_OUTPUT            = bool(os.getenv('APP_DEBUG_OUTPUT'))
    DISCORD_READ_CHANNEL_ID     = str(os.getenv('DISCORD_READ_CHANNEL_ID'))
    DISCORD_WRITE_CHANNEL_ID    = str(os.getenv('DISCORD_WRITE_CHANNEL_ID'))
    DISCORD_BOT_TOKEN           = str(os.getenv('DISCORD_BOT_TOKEN'))
    OLLAMA_MAX_CTX              = int(os.getenv('OLLAMA_MAX_CTX'))
    OLLAMA_RESERVED_OUTPUT      = int(os.getenv('OLLAMA_RESERVED_OUTPUT'))
    SAFE_INPUT_TOKENS = OLLAMA_MAX_CTX - OLLAMA_RESERVED_OUTPUT

    debug_output(f"Starting {APP_NAME} with configuration")
    debug_output(str(config))

    # 1. Go to Discord, read the given channel
    end_time            = datetime.datetime.now(datetime.UTC)
    start_time          = end_time - datetime.timedelta(days=config["days"])
    after_snowflake     = datetime_to_snowflake(start_time)
    before_snowflake    = datetime_to_snowflake(end_time)

    # Delete cache over 30 days old
    remove_cache_after_this_date(
        cache_dir,
        end_time - datetime.timedelta(days=30)
    )

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
    article_links = deduplicate_article_links(db_con, article_links)
    debug_output(f"De-Duplicated Articles to {len(article_links)} total")

    # 2.2 Create the thread manager
    debug_output("Preparing to Lookup Articles, spawning thread manager")
    max_threads = int(os.getenv('MAX_THREADS'))
    thread_manager = ThreadManager(max_concurrent_threads=max_threads)
    for article_dict in article_links:
        # Add threads to the thread Manager
        thread_manager.add_task(
            url=article_dict['url'],
            title=article_dict['title'],
            timeout=int(os.getenv('THREAD_TIMEOUT'))
        )

    # 3. Retrieve the full article content 
    debug_output(f"Running multi-threaded article lookups with max_threads={max_threads}")
    thread_manager.run_all()

    # 4. At this point articles are all stored as 'txt' files, 
    # in the session_cache_dir path. summarize them with Ollama
    # We do this 1 at a time to avoid melting the computer...
    debug_output("Preparing to summarize articles with Ollama")
    summarize_articles(start_time, datetime.datetime.now(datetime.UTC))

    debug_output(f"Generating Final Executive Summary")
    exec_summary = executive_summary(start_time, datetime.datetime.now(datetime.UTC))

    exec_summary_filename = cache_dir / str(end_time.strftime("%Y%m%d%H%M%S")) / 'exec-summary'
    exec_summary_filename.mkdir(parents=True, exist_ok=True)
    exec_summary_final = exec_summary[exec_summary.find("</think>")+8:]
    exec_summary_think = exec_summary[exec_summary.find("<think>")+7:exec_summary.find("</think>")]

    with open(f"{exec_summary_filename}.think.txt", 'w', encoding="utf8") as fhandle:
        fhandle.write(exec_summary_think)
        fhandle.close()

    tts(exec_summary_final, exec_summary_filename)

    # Write The Executive Summary to Disk
    further_reading = get_further_reading_section(start_time, datetime.datetime.now(datetime.UTC))
    exec_summary_final =f"{DOCUMENT_DISCLAIMER}# Executive Summary\n\n" + exec_summary_final + further_reading 
    with open(f'{exec_summary_filename}.md', 'w', encoding="utf8") as fhandle:
        fhandle.write(exec_summary_final)
        fhandle.close()

    debug_output(f"Sending Executive Summary to Discord Channel")
    discord_client.send_attachments([
        f"{exec_summary_filename}.md", f"{exec_summary_filename}.mp3"
    ])

if __name__ == "__main__":
    config = parse_arguments()
    main(config)
    sys.exit(0)