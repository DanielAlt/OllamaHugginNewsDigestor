import subprocess
import torch

from pathlib import Path
from TTS.api import TTS

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