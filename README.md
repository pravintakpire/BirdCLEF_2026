# BirdCLEF+ 2026

Kaggle competition: [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)

**Task**: Identify bird and wildlife species from passive acoustic monitoring soundscapes recorded in the Pantanal wetlands, South America.

**Metric**: Macro-ROC-AUC
**Deadline**: May 27, 2026
**Prize**: $50,000

---

## Setup

```bash
# Install dependencies
uv sync

# Download competition data (requires Kaggle API key in ~/.kaggle/kaggle.json)
bash scripts/download_data.sh
```

---

## Project Structure

```
BirdCLEF_2026/
├── data/
│   ├── raw/          # Raw Kaggle downloads
│   ├── processed/    # Mel-spectrograms / features
│   └── external/     # External data (xeno-canto, eBird)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_submission.ipynb
├── src/
│   ├── config.py     # Paths and hyperparameters
│   ├── audio.py      # Audio loading and mel-spectrogram extraction
│   ├── dataset.py    # PyTorch Dataset
│   ├── models.py     # Model architectures
│   ├── train.py      # Training loop
│   ├── inference.py  # Inference and submission generation
│   └── utils.py      # Metrics and helpers
├── configs/
│   └── baseline.yaml
├── scripts/
│   └── download_data.sh
└── main.py
```

---

## Usage

```bash
# Train
uv run python main.py --mode train --config configs/baseline.yaml

# Inference / generate submission
uv run python main.py --mode infer --config configs/baseline.yaml
```

---

## Results

| Model | CV Macro-ROC-AUC | LB Score | Notes |
|---|---|---|---|
| EfficientNet-B0 baseline | - | - | |
