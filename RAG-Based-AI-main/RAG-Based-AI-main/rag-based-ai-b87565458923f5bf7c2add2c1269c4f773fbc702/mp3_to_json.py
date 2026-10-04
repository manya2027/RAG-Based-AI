import json

import whisper

import config

AUDIOS_DIR = config.AUDIOS_DIR
JSONS_DIR = config.JSONS_DIR


def main():
    if not AUDIOS_DIR.exists():
        print(f"No audio folder found at {AUDIOS_DIR}. Place your .mp3 files there and run the script again.")
        return

    audio_files = sorted(
        file.name for file in AUDIOS_DIR.iterdir() if file.is_file() and file.suffix.lower() in {".mp3", ".wav", ".m4a"}
    )

    if not audio_files:
        print(f"No audio files found in {AUDIOS_DIR}.")
        return

    # Transcription is the slowest step, so never redo a file that already has a json.
    audio_files = [a for a in audio_files if not (JSONS_DIR / f"{a}.json").exists()]
    if not audio_files:
        print("All audio files are already transcribed.")
        return

    JSONS_DIR.mkdir(exist_ok=True)
    model = whisper.load_model(config.WHISPER_MODEL)

    for audio_name in audio_files:
        if "_" in audio_name:
            number = audio_name.split("_")[0]
            title = audio_name.split("_")[1][:-4]
            print(number, title)
            result = model.transcribe(
                audio=str(AUDIOS_DIR / audio_name),
                language="hi",
                task="translate",
                word_timestamps=False,
            )

            chunks = []
            for segment in result["segments"]:
                chunks.append(
                    {
                        "number": number,
                        "title": title,
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                    }
                )

            chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

            out = JSONS_DIR / f"{audio_name}.json"
            tmp = out.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)
            tmp.replace(out)


if __name__ == "__main__":
    main()