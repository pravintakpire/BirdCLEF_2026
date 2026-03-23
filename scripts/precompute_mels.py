"""
Pre-compute mel spectrograms for all training audio and save as .npy files.
Run once before training:
    python scripts/precompute_mels.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import load_config, ROOT
from src.audio import load_audio, make_mel_spectrogram, random_crop, center_crop


def process_file(args):
    filename, audio_dir, out_dir, config = args
    out_path = out_dir / (filename.replace("/", "__").replace(".ogg", ".npy"))
    if out_path.exists():
        return filename, True, "cached"
    try:
        wav, sr = load_audio(str(audio_dir / filename), sr=config["sr"])
        # Save full-length mel (center crop to fixed duration)
        wav = center_crop(wav, sr, config["duration"])
        mel = make_mel_spectrogram(
            wav, sr=sr,
            n_mels=config["n_mels"], n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            fmin=config["fmin"], fmax=config["fmax"],
        )
        np.save(out_path, mel)
        return filename, True, "ok"
    except Exception as e:
        return filename, False, str(e)


def main():
    config = load_config("configs/baseline.yaml")
    df = pd.read_csv(ROOT / config["train_csv"])
    audio_dir = ROOT / config["train_audio_dir"]

    out_dir = ROOT / "data" / "processed" / "mels"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir : {out_dir}")
    print(f"Total files: {len(df)}")

    already_done = sum(1 for _ in out_dir.glob("*.npy"))
    print(f"Already cached: {already_done}")

    args_list = [(row["filename"], audio_dir, out_dir, config) for _, row in df.iterrows()]

    failed = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_file, a): a[0] for a in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing mels"):
            filename, ok, msg = future.result()
            if not ok:
                failed.append((filename, msg))

    print(f"\nDone. Cached: {sum(1 for _ in out_dir.glob('*.npy'))}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f, msg in failed[:10]:
            print(f"  {f}: {msg}")


if __name__ == "__main__":
    main()
