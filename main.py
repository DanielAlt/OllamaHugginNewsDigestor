import sys
import argparse
import datetime
import os

import article_analyzer.core as aa_core
from article_analyzer.discord import datetime_to_snowflake, DiscordAPIClient
from article_analyzer.article_downloader import ThreadManager
from article_analyzer.speech import tts
from article_analyzer.output_formatter import (
    get_further_reading_section,
    DOCUMENT_DISCLAIMER
)
from article_analyzer.article_digestor import (
    deduplicate_article_links,
    summarize_articles,
    executive_summary
)

def parse_arguments() -> dict:
    app_description="""\
An AI (Ollama) digest of a Discord Message Channel devoted to collecting huggin\
 news articles. See the readme.md for more information."""
    parser = argparse.ArgumentParser(
        description=app_description
    )

    parser.add_argument(
        "--days",
        type=int,
        required=True,
        help="Amount of days prior to today to include in the process."
    )

    args = parser.parse_args()
    return vars(args)

def main(config):
    # Installation steps
    aa_core.setup_self()
    aa_core.setup_dotenv()
    system_prompts = aa_core.setup_system_prompts()
    cache_dir = aa_core.setup_cache_dir()
    db_con = aa_core.setup_database()

    aa_core.debug_output(f"Starting {aa_core.APP_NAME} with configuration")
    aa_core.debug_output(str(config))

    # 1. Go to Discord, read the given channel
    end_time            = datetime.datetime.now(datetime.UTC)
    start_time          = end_time - datetime.timedelta(days=config["days"])
    after_snowflake     = datetime_to_snowflake(start_time)
    before_snowflake    = datetime_to_snowflake(end_time)

    # Delete cache over 30 days old
    aa_core.remove_cache_after_this_date(
        cache_dir,
        end_time - datetime.timedelta(days=30)
    )

    aa_core.debug_output("Fetching discord messages...")
    discord_client = DiscordAPIClient(os.getenv("DISCORD_BOT_TOKEN"))
    messages = discord_client.read_messages(int(os.getenv("DISCORD_READ_CHANNEL_ID")), {
        "after": after_snowflake,
        "before": before_snowflake
    })
    aa_core.debug_output(f"Retreived {len(messages)} messages from discord")

    article_links = []
    for message in messages: # Switch to author ID, make configurable
        if message['author']['username'] != "Threat Intelligence Bot":
            continue
        if message['embeds']:
            article_links.append({
                "url": message['embeds'][0]['url'],
                "title": message['embeds'][0]['title']
            })

    aa_core.debug_output(f"Filtered to {len(article_links)} articles")
    article_links = deduplicate_article_links(article_links)
    aa_core.debug_output(f"De-Duplicated Articles to {len(article_links)} total")

    # 2.2 Create the thread manager
    aa_core.debug_output("Preparing to Lookup Articles, spawning thread manager")
    thread_manager = ThreadManager(max_concurrent_threads=int(os.getenv('MAX_THREADS')))
    for article_dict in article_links:
        # Add threads to the thread Manager
        thread_manager.add_task(
            url=article_dict['url'],
            title=article_dict['title'],
            timeout=int(os.getenv('THREAD_TIMEOUT'))
        )

    # 3. Retrieve the full article content 
    aa_core.debug_output(f"Running multi-threaded article lookups with max_threads={os.getenv('MAX_THREADS')}")
    thread_manager.run_all()

    # 4. At this point articles are all stored as 'txt' files, 
    # in the session_cache_dir path. summarize them with Ollama
    # We do this 1 at a time to avoid melting the computer...
    aa_core.debug_output("Preparing to summarize articles with Ollama")
    summarize_articles(
        start_time, 
        datetime.datetime.now(datetime.UTC), 
        system_prompts['article_summary_prompt']
    )

    aa_core.debug_output(f"Generating Final Executive Summary")
    exec_summary = executive_summary(
        start_time, 
        datetime.datetime.now(datetime.UTC),
        system_prompts['executive_summary_prompt']
    )

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

    aa_core.debug_output(f"Sending Executive Summary to Discord Channel")
    discord_client.send_attachments(
        os.getenv("DISCORD_WRITE_CHANNEL_ID"), 
        [ f"{exec_summary_filename}.md", f"{exec_summary_filename}.mp3"]
    )

if __name__ == "__main__":
    config = parse_arguments()
    main(config)
    sys.exit(0)