import numpy as np
import librosa


def load_audio(path: str, sr: int = 32000) -> tuple[np.ndarray, int]:
    waveform, _ = librosa.load(path, sr=sr, mono=True)
    return waveform, sr


def random_crop(waveform: np.ndarray, sr: int, duration: float) -> np.ndarray:
    target_len = int(sr * duration)
    if len(waveform) <= target_len:
        # Pad with zeros if shorter than target
        return np.pad(waveform, (0, target_len - len(waveform)))
    start = np.random.randint(0, len(waveform) - target_len)
    return waveform[start : start + target_len]


def center_crop(waveform: np.ndarray, sr: int, duration: float) -> np.ndarray:
    target_len = int(sr * duration)
    if len(waveform) <= target_len:
        return np.pad(waveform, (0, target_len - len(waveform)))
    start = (len(waveform) - target_len) // 2
    return waveform[start : start + target_len]


def make_mel_spectrogram(
    waveform: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 320,
    fmin: float = 20.0,
    fmax: float = 16000.0,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return mel_db.astype(np.float32)


def sliding_windows(
    waveform: np.ndarray, sr: int, duration: float, step: float
) -> list[np.ndarray]:
    target_len = int(sr * duration)
    step_len = int(sr * step)
    windows = []
    start = 0
    while start + target_len <= len(waveform):
        windows.append(waveform[start : start + target_len])
        start += step_len
    if not windows:
        windows.append(np.pad(waveform, (0, target_len - len(waveform))))
    return windows
