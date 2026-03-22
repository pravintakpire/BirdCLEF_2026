#!/usr/bin/env bash
# Download BirdCLEF 2026 competition data
# Requires: kaggle CLI configured with ~/.kaggle/kaggle.json

set -e

mkdir -p data/raw
kaggle competitions download -c birdclef-2026 -p data/raw/
cd data/raw && unzip -q birdclef-2026.zip && rm birdclef-2026.zip
echo "Data downloaded and extracted to data/raw/"
