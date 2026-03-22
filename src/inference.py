from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.audio import load_audio, sliding_windows, make_mel_spectrogram


@torch.no_grad()
def predict_soundscape(
    model: nn.Module,
    audio_path: str,
    species_list: list[str],
    config: dict,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    sr = config["sr"]
    duration = config["duration"]
    step = config.get("inference_step", duration)

    waveform, _ = load_audio(audio_path, sr=sr)
    windows = sliding_windows(waveform, sr, duration, step)

    preds = []
    for window in windows:
        mel = make_mel_spectrogram(
            window,
            sr=sr,
            n_mels=config["n_mels"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            fmin=config["fmin"],
            fmax=config["fmax"],
        )
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(mel_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds.append(probs)

    preds_array = np.array(preds)  # (num_windows, num_classes)

    # Build per-5s-chunk rows for submission
    rows = []
    stem = Path(audio_path).stem
    for i, prob_row in enumerate(preds_array):
        row = {"row_id": f"{stem}_{int((i + 1) * step)}"}
        for j, species in enumerate(species_list):
            row[species] = prob_row[j]
        rows.append(row)

    return pd.DataFrame(rows)


def generate_submission(
    model: nn.Module,
    test_soundscapes: list[str],
    species_list: list[str],
    config: dict,
    device: torch.device,
    output_path: Path,
) -> None:
    all_dfs = []
    for path in tqdm(test_soundscapes, desc="Inference"):
        df = predict_soundscape(model, path, species_list, config, device)
        all_dfs.append(df)

    submission = pd.concat(all_dfs, ignore_index=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
