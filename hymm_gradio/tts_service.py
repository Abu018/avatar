import os
import uuid
import subprocess

from deepgram import DeepgramClient
from dotenv import load_dotenv

load_dotenv()

TEMP_DIR = "./temp"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")


def _ensure_temp_dir():
    os.makedirs(TEMP_DIR, exist_ok=True)


def synthesize_to_wav(
    text: str,
    voice_id: str = "aura-2-thalia-en",
    language: str = None,
    speed: float = 1.0,
    provider: str = "deepgram",
) -> str:
    if provider not in (None, "deepgram"):
        raise ValueError(f"Unsupported provider: {provider}")
    if not text or not text.strip():
        raise ValueError("text is required")
    if DEEPGRAM_API_KEY is None:
        raise ValueError("DEEPGRAM_API_KEY is not set")

    _ensure_temp_dir()
    audio_id = str(uuid.uuid4())
    output_mp3 = os.path.join(TEMP_DIR, f"{audio_id}.mp3")
    output_wav = os.path.join(TEMP_DIR, f"{audio_id}.wav")

    deepgram = DeepgramClient()
    response = deepgram.speak.v1.audio.generate(
        text=text,
        model=voice_id or "aura-2-thalia-en",
    )

    with open(output_mp3, "wb") as audio_file:
        audio_file.write(response.stream.getvalue())

    # Convert to WAV so downstream audio preprocessing is consistent.
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        output_mp3,
        "-ac",
        "1",
        "-ar",
        "16000",
        output_wav,
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Keep temp directory clean.
    if os.path.exists(output_mp3):
        os.remove(output_mp3)

    return output_wav

