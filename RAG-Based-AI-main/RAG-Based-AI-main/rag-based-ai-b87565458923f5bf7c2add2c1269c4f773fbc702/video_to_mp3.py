# Converts the videos to mp3, several at a time, skipping ones already converted
import subprocess
from concurrent.futures import ThreadPoolExecutor

import config


def target_name(file):
    # e.g. "Your First HTML Website ｜ Sigma Web Development Course - Tutorial #2 [abc].mp4"
    tutorial_number = file.split(" [")[0].split(" #")[1]
    file_name = file.split(" ｜ ")[0]
    return f"{tutorial_number}_{file_name}.mp3"


def convert(file):
    try:
        name = target_name(file)
    except IndexError:
        return f"skipped (unrecognised name): {file}"
    out = config.AUDIOS_DIR / name
    if out.exists():
        return f"already converted: {name}"
    tmp = out.with_suffix(".part.mp3")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(config.VIDEOS_DIR / file), "-vn", str(tmp)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        return f"FAILED {file}: {result.stderr.strip()}"
    tmp.replace(out)
    return f"converted: {name}"


def main():
    config.AUDIOS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in config.VIDEOS_DIR.iterdir() if p.is_file())
    with ThreadPoolExecutor(max_workers=config.FFMPEG_WORKERS) as pool:
        for message in pool.map(convert, files):
            print(message)


if __name__ == "__main__":
    main()
