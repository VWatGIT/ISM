import sys
from pathlib import Path
import subprocess


def extract_all_frames(root: Path, SAVE_DIR: str):
    VIDEO_EXTS = {'.avi'}
    
    for vid in root.rglob('*'):
        if vid.is_file() and vid.suffix.lower() in VIDEO_EXTS:

            outdir = Path(SAVE_DIR) / f"{vid.stem}_frames"
            outdir.mkdir(parents=True, exist_ok=True)

            out_pattern = str(outdir / "%03d.bmp")

            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(vid), "fps=1", "-r", "1", out_pattern]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print("ffmpeg failed for", vid)


if __name__ == "__main__":
    root = Path.cwd()
    print(root)

    SAVE_DIR = r"C:\Users\Valentin\Documents\GIT_REPS\TUHH\ISM\Data\Images"

    extract_all_frames(root, SAVE_DIR=SAVE_DIR)
