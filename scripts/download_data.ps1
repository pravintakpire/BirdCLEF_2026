# Download BirdCLEF 2026 competition data
# Requires: KAGGLE_TOKEN set in .env file at project root

$ErrorActionPreference = "Stop"

# Load .env from project root
$envFile = Join-Path $PSScriptRoot "..\..env" | Resolve-Path -ErrorAction SilentlyContinue
if (-not $envFile) { $envFile = Join-Path $PSScriptRoot "..\.env" }

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
    Write-Host "Loaded environment from .env"
} else {
    Write-Error ".env file not found. Create one with KAGGLE_TOKEN=your_token"
}

New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
kaggle competitions download -c birdclef-2026 -p data\raw\
Expand-Archive -Path "data\raw\birdclef-2026.zip" -DestinationPath "data\raw\" -Force
Remove-Item "data\raw\birdclef-2026.zip"
Write-Host "Data downloaded and extracted to data\raw\"
