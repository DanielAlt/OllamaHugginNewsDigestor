import os
import sqlite3
import sys
import shutil
import platform
import datetime

from pathlib import Path
from dotenv import load_dotenv

APP_NAME="OllamaHugginBridge"
MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"

def debug_output(message, error=False):
    if not os.getenv("APP_DEBUG_OUTPUT"): 
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
        env_example_path = DATA_DIR / ".env.example"
        with open(env_example_path, 'r', encoding="utf8") as fhandle1:
            env_example = fhandle1.read()

        with open(env, 'w', encoding='utf8') as fhandle2:
            fhandle2.write(env_example)

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

def load_default_prompts():
    article_summary_prompt_path = DATA_DIR / "article-summary-prompt.md"
    executive_summary_prompt_path = DATA_DIR / "executive-summary-prompt.md"

    article_summary_prompt = ""
    with open(article_summary_prompt_path, 'r', encoding="utf8") as fhandle:
        article_summary_prompt = fhandle.read()

    executive_summary_prompt = ""
    with open(executive_summary_prompt_path, 'r', encoding="utf8") as fhandle:
        executive_summary_prompt = fhandle.read()
    
    return ({
        "executive_summary_prompt": executive_summary_prompt,
        "article_summary_prompt": article_summary_prompt 
    })

def setup_system_prompts():
    default_prompts = load_default_prompts()

    base_path = get_installation_path()
    summary_prompt_path = base_path / "article-summary-prompt.md"
    executive_prompt_path = base_path / "executive-summary-prompt.md"

    if summary_prompt_path.is_file():
        with open(summary_prompt_path, 'r', encoding="utf8") as fhandle1:
            default_prompts['article_summary_prompt'] = fhandle1.read()
    else:
        with open(summary_prompt_path, 'w', encoding="utf8") as fhandle1:
            fhandle1.write(default_prompts['article_summary_prompt'])
    
    if executive_prompt_path.is_file():
        with open(executive_prompt_path, 'r', encoding="utf8") as fhandle2:
            default_prompts['executive_summary_prompt'] = fhandle2.read()
    else:
        with open(executive_prompt_path, 'w', encoding="utf8") as fhandle2:
            fhandle2.write(default_prompts['executive_summary_prompt'])

    return default_prompts
            
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