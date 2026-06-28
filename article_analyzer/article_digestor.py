import os
import datetime
import json 

import tiktoken

from ollama import chat, ChatResponse
from pydantic import BaseModel, Field, constr
from typing import Literal

from .core import get_db_connection, debug_output

def get_safe_input_tokens():
    return int(os.getenv("OLLAMA_MAX_CTX")) - int(os.getenv("OLLAMA_RESERVED_OUTPUT"))

def deduplicate_article_links(article_links) -> list:
    db_con = get_db_connection()
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

def truncate_to_token_limit(encoding, text, max_tokens):
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    truncated = encoding.decode(tokens[:max_tokens])

    return truncated


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


def summarize_articles(start_time, end_time, system_prompt):
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

        if len(encoding.encode(system_prompt + article_content_raw)) > get_safe_input_tokens():
            article_content_raw = truncate_to_token_limit(
                article_content,
                get_safe_input_tokens() - len(encoding.encode(system_prompt))
            )
        
        debug_output(f"Summarizing Article: {article_title}")
        now = datetime.datetime.now()
        response: ChatResponse = chat(
            model=os.getenv('OLLAMA_MODEL_NAME'), 
            messages=[
                {'role': 'system', 'content': system_prompt},
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

def executive_summary(start_time, end_time, system_prompt) -> str:
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

    if len(encoding.encode(system_prompt + article_summaries)) > get_safe_input_tokens():
        article_summaries = truncate_to_token_limit(
            article_summaries,
            get_safe_input_tokens() - len(encoding.encode(system_prompt))
        )
        
    now = datetime.datetime.now()
    response: ChatResponse = chat(
        model=os.getenv("OLLAMA_MODEL_NAME"), 
        messages=[
            { 'role': 'system', 'content': system_prompt }, 
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
