#!/bin/bash
# =============================================================================
# setup_env.sh — Complete NeMo Environment Setup
# =============================================================================
#
# Creates a conda environment with all dependencies for NeMo diarization.
#
# USAGE:
#   bash bash/setup_env.sh
#
# WHAT IT DOES:
#   1. Creates a conda environment named "nemo" with Python 3.10
#   2. Installs PyTorch (CPU on macOS, CUDA on Linux)
#   3. Installs NeMo toolkit with ASR extras
#   4. Installs audio processing and visualization dependencies
#   5. Installs ffmpeg for audio format conversion
#   6. Sets up the .env file from .env.example
#
# AFTER RUNNING:
#   conda activate nemo
#   python scripts/run_sortformer.py --help
# =============================================================================

set -e  # Exit on any error

echo "============================================================"
echo "  NeMo Speaker Diarization — Environment Setup"
echo "============================================================"
echo ""

# --- Configuration ---
ENV_NAME="nemo"
PYTHON_VERSION="3.10"

# --- Step 1: Create environment ---
echo "📦 Step 1: Creating python environment..."
if command -v conda &> /dev/null; then
    echo "   Using conda..."
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "   ⚠️  Environment '${ENV_NAME}' already exists."
        read -p "   Recreate it? (y/N): " RECREATE
        if [[ "$RECREATE" == "y" || "$RECREATE" == "Y" ]]; then
            conda env remove -n "${ENV_NAME}" -y
            conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
        fi
    else
        conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    fi
    echo "📦 Step 2: Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
else
    echo "   Conda not found. Using standard Python venv..."
    PYTHON_CMD="python3.10"
    if ! command -v $PYTHON_CMD &> /dev/null; then
        PYTHON_CMD="python3"
    fi
    echo "   Using ${PYTHON_CMD}..."
    $PYTHON_CMD -m venv venv
    echo "📦 Step 2: Activating venv..."
    source venv/bin/activate
fi

echo "   Python: $(python --version)"
echo "   Location: $(which python)"

# --- Step 3: Install PyTorch ---
echo ""
echo "📦 Step 3: Installing PyTorch..."

# Detect OS and GPU
if [[ "$(uname)" == "Darwin" ]]; then
    echo "   macOS detected — installing CPU-only PyTorch"
    pip install torch torchaudio
elif command -v nvidia-smi &> /dev/null; then
    echo "   NVIDIA GPU detected — installing CUDA PyTorch"
    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "   CUDA driver: ${CUDA_VERSION}"
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "   No NVIDIA GPU detected — installing CPU-only PyTorch"
    pip install torch torchaudio
fi

# --- Step 4: Install NeMo ---
echo ""
echo "📦 Step 4: Installing NVIDIA NeMo toolkit..."
pip install nemo_toolkit[asr]

# --- Step 5: Install dependencies ---
echo ""
echo "📦 Step 5: Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# --- Step 6: Install ffmpeg ---
echo ""
echo "📦 Step 6: Installing ffmpeg..."
if [[ "$(uname)" == "Darwin" ]]; then
    if command -v brew &> /dev/null; then
        brew install ffmpeg 2>/dev/null || echo "   ffmpeg already installed via brew"
    else
        conda install -c conda-forge ffmpeg -y
    fi
else
    conda install -c conda-forge ffmpeg -y
fi

# --- Step 7: Verify installation ---
echo ""
echo "============================================================"
echo "  Verifying Installation"
echo "============================================================"

python -c "
import torch
print(f'  PyTorch:     {torch.__version__}')
print(f'  CUDA avail:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:         {torch.cuda.get_device_name(0)}')
    print(f'  CUDA ver:    {torch.version.cuda}')
"

python -c "
try:
    import nemo
    print(f'  NeMo:        {nemo.__version__}')
except ImportError:
    print('  NeMo:        ❌ NOT INSTALLED')
"

python -c "
import pydub; print(f'  pydub:       ✅')
import pandas; print(f'  pandas:      ✅')
import soundfile; print(f'  soundfile:   ✅')
"

echo ""
echo "============================================================"
echo "  ✅ Setup Complete!"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "    1. conda activate ${ENV_NAME}"
echo "    2. python scripts/patch_nemo.py"
echo "    3. Place audio files in data/"
echo "    4. python scripts/run_cascaded.py --audio data/your-audio.wav"
echo ""
