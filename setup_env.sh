#!/usr/bin/env bash
# Create the conda environment used by ML4OCRP and install its Python packages.
# From ML4OCRP/:
#   bash setup_env.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="ocrp"
PYTHON_VERSION="3.13"

if ! command -v conda >/dev/null 2>&1; then
  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh" \
    "/opt/anaconda3/etc/profile.d/conda.sh" \
    "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
  do
    if [[ -f "$candidate" ]]; then
      # shellcheck disable=SC1090
      source "$candidate"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Anaconda or Miniconda, then open a new terminal." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists. Packages will be installed into it."
else
  echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION."
  conda create --name "$ENV_NAME" "python=$PYTHON_VERSION" --yes
fi

echo "Installing packages from requirements.txt."
conda run --name "$ENV_NAME" python -m pip install --upgrade pip
conda run --name "$ENV_NAME" python -m pip install -r "$ROOT/requirements.txt"

mkdir -p \
  "$ROOT/Data/Original" \
  "$ROOT/Data/Preprocessed" \
  "$ROOT/Data/Model" \
  "$ROOT/Data/Predict"

echo
echo "Environment '$ENV_NAME' is ready."
echo "Activate it in this terminal before training:"
echo "  conda activate $ENV_NAME"
echo "  export KMP_DUPLICATE_LIB_OK=TRUE"
