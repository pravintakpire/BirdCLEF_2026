from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.audio import load_audio, random_crop, center_crop, make_mel_spectrogram


class BirdCLEFDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        species_list: list[str],
        config: dict,
        train: bool = True,
        mel_cache_dir: Path | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.config = config
        self.train = train
        self.mel_cache_dir = Path(mel_cache_dir) if mel_cache_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_mel(self, filename: str) -> np.ndarray:
        # Fast path: load pre-computed .npy cache
        if self.mel_cache_dir is not None:
            cache_path = self.mel_cache_dir / (filename.replace("/", "__").replace(".ogg", ".npy"))
            if cache_path.exists():
                return np.load(cache_path)

        # Slow path: load audio and compute on the fly
        sr = self.config["sr"]
        duration = self.config["duration"]
        waveform, _ = load_audio(str(self.audio_dir / filename), sr=sr)
        if self.train:
            waveform = random_crop(waveform, sr, duration)
        else:
            waveform = center_crop(waveform, sr, duration)
        return make_mel_spectrogram(
            waveform, sr=sr,
            n_mels=self.config["n_mels"], n_fft=self.config["n_fft"],
            hop_length=self.config["hop_length"],
            fmin=self.config["fmin"], fmax=self.config["fmax"],
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        mel = self._load_mel(row["filename"])
        # Shape: (1, n_mels, time) for single-channel CNN input
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        primary = row.get("primary_label", None)
        if primary and primary in self.species_to_idx:
            label[self.species_to_idx[primary]] = 1.0
        secondary = row.get("secondary_labels", [])
        if isinstance(secondary, str):
            secondary = secondary.split()
        for s in secondary:
            if s in self.species_to_idx:
                label[self.species_to_idx[s]] = 1.0

        return mel_tensor, label
