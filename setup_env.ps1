# Create the conda environment used by ML4OCRP and install its Python packages.
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvName = "ocrp"
$PythonVersion = "3.13"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found on PATH. Install Anaconda or Miniconda, then open a new terminal."
}

$listed = conda env list
$exists = $listed -match "(?m)^\s*\*?\s*$EnvName\s"
if ($exists) {
    Write-Host "Conda environment '$EnvName' already exists. Packages will be installed into it."
} else {
    Write-Host "Creating conda environment '$EnvName' with Python $PythonVersion."
    conda create --name $EnvName "python=$PythonVersion" --yes
    if ($LASTEXITCODE -ne 0) {
        throw "conda create failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Installing packages from requirements.txt."
conda run --name $EnvName python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed with exit code $LASTEXITCODE."
}
conda run --name $EnvName python -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "pip install failed with exit code $LASTEXITCODE."
}

$DataDirs = @(
    "Data\Original",
    "Data\Preprocessed",
    "Data\Model",
    "Data\Predict"
)
foreach ($relative in $DataDirs) {
    $path = Join-Path $Root $relative
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

Write-Host ""
Write-Host "Environment '$EnvName' is ready."
Write-Host "Activate it in this terminal, then set the OpenMP variable before training:"
Write-Host "  conda activate $EnvName"
Write-Host '  $env:KMP_DUPLICATE_LIB_OK = "TRUE"'
