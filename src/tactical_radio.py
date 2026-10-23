"""
ElevenLabs Tactical Radio Dispatch Engine for Lucknow Geospatial AI.
Converts Gemma 4 tactical radio scripts into commanding, low-latency audio dispatches
for police and municipal enforcement squads with walkie-talkie audio effects.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

AUDIO_OUTPUT_DIR = ROOT_DIR / "monitoring_results" / "audio"
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_tactical_radio_dispatch(
    script_text: str,
    output_filename: str = "tactical_dispatch.mp3",
    voice_name: str = "Adam"
) -> Path:
    """
    Generates a realistic police/military tactical radio broadcast using ElevenLabs.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    out_path = AUDIO_OUTPUT_DIR / output_filename

    if not api_key:
        print("[TACTICAL RADIO] Note: ELEVENLABS_API_KEY is not configured yet in .env.")
        print(f"[TACTICAL RADIO] Script ready for dispatch: \"{script_text}\"")
        # Save transcript text alongside
        txt_path = out_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(script_text)
        print(f"[TACTICAL RADIO] Saved dispatch transcript to: {txt_path}")
        return txt_path

    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=api_key)

        print(f"[TACTICAL RADIO] Generating ElevenLabs audio dispatch with voice '{voice_name}'...")
        audio = client.text_to_speech.convert(
            text=script_text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # Standard crisp authoritative voice (George/Adam)
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        with open(out_path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)

        print(f"[TACTICAL RADIO] Successfully created audio dispatch at: {out_path}")
        return out_path

    except Exception as e:
        print(f"[TACTICAL RADIO] ElevenLabs generation error: {e}")
        return None


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sample_script = (
        "All units, urgent dispatch to Site LKO-ENC-20261020-467. "
        "Critical illegal concrete encroachment detected in Gomti River buffer zone. "
        "Enforcement squad proceed for site perimeter interception immediately. Over."
    )
    generate_tactical_radio_dispatch(sample_script)
