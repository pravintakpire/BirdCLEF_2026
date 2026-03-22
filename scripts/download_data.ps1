# Download BirdCLEF 2026 competition data
# Requires: KAGGLE_API_TOKEN set in .env file at project root

$ErrorActionPreference = "Stop"

# Load .env from project root
$envFile = Join-Path $PSScriptRoot "..\..env"
if (-not (Test-Path $envFile)) { $envFile = Join-Path $PSScriptRoot "..\.env" }

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -match "^([^#][^=]*)=(.*)$") {
            $key = $matches[1].Trim()
            $val = $matches[2].Trim().TrimEnd("`r")
            [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
            Set-Item -Path "Env:\$key" -Value $val
        }
    }
    Write-Host "Loaded environment from .env"
} else {
    Write-Error ".env file not found. Create one with KAGGLE_API_TOKEN=your_token"
}

Write-Host "KAGGLE_API_TOKEN is set: $($env:KAGGLE_API_TOKEN -ne $null -and $env:KAGGLE_API_TOKEN -ne '')"

New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
uv run kaggle competitions download -c birdclef-2026 -p data\raw\
Expand-Archive -Path "data\raw\birdclef-2026.zip" -DestinationPath "data\raw\" -Force
Remove-Item "data\raw\birdclef-2026.zip"
Write-Host "Data downloaded and extracted to data\raw\"
