import datetime
import requests
import time

from contextlib import ExitStack

DISCORD_EPOCH = 1420070400000

def datetime_to_snowflake(dt: datetime.datetime) -> int:
    unix_ms = int(dt.timestamp() * 1000)
    return (unix_ms - DISCORD_EPOCH) << 22

class DiscordAPIClient():
    """ 
        Discord API Client Class, for Reading and Writing Messages to 
        the NewsFeed channel
    """
    def __init__(self, discord_token):
        self.headers = { "Authorization": f"Bot {discord_token}" }

    def read_messages(self, channel_id, params: dict) -> dict:
        api_url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        """ Read Messages from the hardcoded channel """
        response = requests.get(
            api_url, 
            headers=self.headers, 
            params=params
        )
        messages = response.json()
        return messages

    def send_message(self, channel_id, content: str, retry_count: int = 0) -> dict:

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
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
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

    def send_attachments(self, channel_id, attachments):
        
        with ExitStack() as stack:
            files = []
            
            for i, filename in enumerate(attachments):
                f = stack.enter_context(open(filename, "rb"))
                files.append((f"files[{i}]", f))

            response = requests.post(
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
                data={"content": "Your files, as requested."}, 
                files=files, 
                headers=self.headers
            )