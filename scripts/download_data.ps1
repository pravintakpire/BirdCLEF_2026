# Download BirdCLEF 2026 competition data
# Requires: kaggle CLI configured with %USERPROFILE%\.kaggle\kaggle.json

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
kaggle competitions download -c birdclef-2026 -p data\raw\
Expand-Archive -Path "data\raw\birdclef-2026.zip" -DestinationPath "data\raw\" -Force
Remove-Item "data\raw\birdclef-2026.zip"
Write-Host "Data downloaded and extracted to data\raw\"
