# Ollama Huggin Bridge

## Overview

Ollama Huggin Bridge is a cybersecurity news automation tool.

It reads article links from a Discord channel, downloads the articles, summarizes them with a local Ollama model, creates an executive briefing, generates a narrated audio version using XTTS voice cloning, and posts the results back to Discord.

The project is designed to help security teams quickly review large numbers of threat intelligence and cybersecurity news articles without reading every source manually.

## What the Application Does

1. Reads messages from a Discord channel containing cybersecurity news articles.
2. Extracts article URLs from Discord embeds.
3. Downloads and stores the article content locally.
4. Uses a local Ollama model to create structured summaries.
5. Extracts:
   - Organizations and vendors
   - Malware names
   - Indicators of Compromise (IOCs)
   - Severity information
6. Stores article summaries as JSON files.
7. Generates an executive summary covering all articles.
8. Converts the executive summary into speech using XTTS.
9. Produces:
   - Executive summary text file
   - Executive summary MP3 file
10. Uploads the final report and audio briefing back to Discord.

## Architecture

Discord News Feed
        |
        v
Download Articles
        |
        v
Extract Text Content
        |
        v
Ollama Summarization
        |
        v
Structured JSON Summaries
        |
        v
Executive Summary
        |
        +----> Text Report
        |
        +----> XTTS Audio Report
        |
        v
Post Results to Discord

## Requirements

### Software

- Python 3.11
- Ollama
- FFmpeg
- Discord Bot Token
- CUDA-capable GPU (recommended for XTTS)

### Python Packages

The project uses packages such as:

- requests
- ollama
- tiktoken
- beautifulsoup4
- python-dotenv
- pydantic
- torch
- TTS

Install all dependencies through `requirements.txt`.

## Installation

### 1. Install Python

Install Python 3.11.

### 2. Create a Virtual Environment

```bash
py -3.11 -m venv venv
```

### 3. Activate the Environment

Windows:

```bash
.\venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 4. Upgrade Pip

```bash
python -m pip install --upgrade pip
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Install FFmpeg

Windows:

```bash
winget install "FFmpeg (Essentials Build)"
```

Verify installation:

```bash
ffmpeg -version
```

### 7. Install Ollama

Start Ollama and pull a model:

```bash
ollama run qwen3:4b
```

You may substitute another compatible model if desired.

## Configuration

Create a `.env` file in the project directory.

Example:

```env
APP_NAME=OllamaHugginBridge
APP_VERSION=0.0.1
APP_CACHE_DIR=cache
APP_DEBUG_OUTPUT=False

DISCORD_READ_CHANNEL_ID=<source-channel-id>
DISCORD_WRITE_CHANNEL_ID=<destination-channel-id>
DISCORD_BOT_TOKEN=<discord-bot-token>

OLLAMA_MAX_CTX=16384
OLLAMA_RESERVED_OUTPUT=1024
```

### Discord Permissions

The bot requires permission to:

- Read messages
- Read message history
- Send messages
- Upload files

## Voice Cloning

The application expects a reference voice sample at:

```text
voice-samples/voice-sample3.1m.wav
```

Replace this file with your preferred speaker sample.

The XTTS model used is:

```text
tts_models/multilingual/multi-dataset/xtts_v2
```

## Running the Application

Process articles from the last day:

```bash
python main.py --days 1
```

Additional options:

```bash
python main.py ^
  --days 7 ^
  --max-threads 8 ^
  --thread-timeout 60 ^
  --model-name qwen3:4b
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| --days | Number of days of Discord history to process |
| --max-threads | Maximum concurrent article download threads |
| --thread-timeout | Timeout for article downloads |
| --model-name | Ollama model name |

## Output

For each run, the application creates a timestamped cache directory containing:

```text
cache/
└── YYYYMMDDHHMMSS/
    ├── articles/
    ├── summaries/
    ├── exec-summary.txt
    ├── exec-summary.think.txt
    └── exec-summary.mp3
```

### Generated Files

- Article text cache
- Structured article summaries (JSON)
- Executive summary report
- XTTS-generated audio briefing
- AI reasoning output (if present)

## Known Limitations

Some websites load article content dynamically using JavaScript frameworks such as React.

In those cases, the current implementation may retrieve incomplete content because it relies primarily on the `requests` library.

A future enhancement would be integrating a headless browser solution to render pages before extraction.

## Intended Use Case

This project is intended for:

- Security Operations Centers (SOC)
- Threat Intelligence Teams
- Cybersecurity Analysts
- Incident Responders
- Security Managers who require executive briefings
