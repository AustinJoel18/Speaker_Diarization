#!/usr/bin/env python3
"""
=============================================================================
run_cascaded.py — Cascaded Speaker Diarization Pipeline
=============================================================================

This script runs the TRADITIONAL cascaded diarization pipeline:

  Audio → VAD (MarbleNet) → Speaker Embeddings (TitaNet) → Clustering → Segments

Unlike Sortformer (end-to-end), the cascaded approach uses SEPARATE models
for each stage. The big advantage: you can SWAP each model independently.

MODEL SWAP POINTS (change these in the YAML config or via CLI):
  --vad-model         → Voice Activity Detection (e.g., vad_marblenet)
  --embedding-model   → Speaker Embeddings (e.g., titanet_large, titanet_small)
  --num-speakers      → Number of speakers (0 = auto-detect)

USAGE:
  # Default pipeline:
  python scripts/run_cascaded.py --audio data/toy-example.wav

  # Swap embedding model:
  python scripts/run_cascaded.py --audio data/toy-example.wav \
      --embedding-model titanet_small

  # Swap VAD model:
  python scripts/run_cascaded.py --audio data/toy-example.wav \
      --vad-model vad_multilingual_marblenet

  # With config file:
  python scripts/run_cascaded.py --config configs/diarization/diar_infer_general.yaml

OFFICIAL DOCS:
  https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html
=============================================================================
"""

import os
import sys
import time
import argparse
import logging
import json
import multiprocessing
from pathlib import Path

# Fix macOS multiprocessing PicklingError
# MacOS defaults to 'spawn' for multiprocessing which can cause issues with NeMo's multiprocessing. Setting it to 'fork' can resolve these issues.
# Darwin = macOS, the OS developed by Apple.
if sys.platform == "darwin":
    multiprocessing.set_start_method("fork", force=True)

# Configure logging format and level
# Minimum log level to display is INFO (DEBUG logs not shown)
# Level names are INFO, WARNING, ERROR, CRITICAL.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Pre-processing 1: Choose to run diarization on CUDA (NVIDIA Driver) GPU or CPU. Default is "auto" which will use GPU if available.
def detect_device(requested: str = "auto") -> str:
    """Auto-detect best available compute device."""
    import torch
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠️  CUDA not available, falling back to CPU")
            device = "cpu"
    logger.info(f"🖥️  Using device: {device}")
    return device


# Pre-processing 2: Ensure the audio file is in a format that NeMo can reliably handle (Mono WAV).
def preprocess_audio(audio_path: str) -> str:
    """
    Automatically converts any input audio (mp3, stereo wav, etc.) to 
    Mono (1-channel) WAV format, which is strictly required by NeMo.
    """
    # AudioSegment is a Python representation of an audio file
    from pydub import AudioSegment
    
    # Convert a (possibly) relative audio path to an absolute path for consistency, and convert to string for pydub compatibility
    audio_path = str(Path(audio_path).resolve())
    logger.info(f"🔍 Checking audio file: {audio_path}")
    
    try:
        # load audio file as an AudioSegment object
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        logger.error(f"❌ Failed to load audio file: {e}")
        sys.exit(1)
        
    if audio.channels == 1 and audio_path.lower().endswith(".wav"):
        logger.info("✅ Audio is already Mono WAV. No conversion needed.")
        return audio_path
        
    # Needs conversion!
    # Create new filename by appending "_mono" before the extension. Eg: "audio.mp3" -> "audio_mono.wav", where base = "audio" and ext = ".mp3"
    base, ext = os.path.splitext(audio_path)
    new_path = f"{base}_mono.wav"
    logger.info(f"⚠️  Converting {ext} to Mono WAV -> {new_path}")
    
    # Force 1 channel
    audio = audio.set_channels(1)
    # Export to WAV
    audio.export(new_path, format="wav")
    
    logger.info("✅ Conversion complete!")
    return new_path

# Pre-processing 3: Create a NeMo-style manifest file for the audio input.
def create_manifest(audio_path: str, manifest_path: str, num_speakers=None):
    """
    Create a NeMo-style manifest file for the audio input.

    NeMo's ClusteringDiarizer expects a JSON-lines manifest file where
    each line describes one audio file:
      {"audio_filepath": "/path/to/audio.wav", "offset": 0, "duration": null,
       "label": "infer", "text": "-", "num_speakers": null, "rttm_filepath": null}
    """
    # Resolve the audio path to an absolute path for consistency, and convert to string for JSON compatibility
    audio_path = str(Path(audio_path).resolve())

    # Get audio duration
    try:
        import soundfile as sf
        # Extract an object containing metadata about the audio file, including duration.
        info = sf.info(audio_path)
        # Extract the duration in seconds from the info object.
        duration = info.duration
    except Exception:
        duration = None  # Let NeMo figure it out

    # Create manifest entry. This is the format NeMo expects for each audio file.
    entry = {
        "audio_filepath": audio_path,
        "offset": 0,
        "duration": duration,
        "label": "infer",
        "text": "-",
        "num_speakers": num_speakers,
        "rttm_filepath": None,
    }

    with open(manifest_path, "w") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(f"📝 Manifest created: {manifest_path}")
    return manifest_path

# Main function to run the cascaded diarization pipeline with configurable models and parameters.
# Default models: VAD = "vad_marblenet", Embeddings = "titanet_large". These can be swapped via CLI or YAML config.
def run_cascaded_pipeline(
    audio_path: str,
    output_dir: str,
    vad_model: str = "vad_marblenet",
    embedding_model: str = "titanet_large",
    msdd_model: str = None,
    num_speakers: int = 0,
    max_num_speakers: int = 8, # TODO: Check if max_num_speakers needs to be set manually and there is a limit.
    device: str = "cpu",
):
    """
    Run the cascaded diarization pipeline.

    PIPELINE STAGES:
    ┌─────────┐    ┌──────────────┐    ┌────────────┐    ┌───────────┐
    │  Audio  │ →  │  VAD         │ →  │  Speaker   │ →  │ Clustering│ → Results
    │  Input  │    │ (MarbleNet)  │    │ Embeddings │    │ (Spectral)│
    └─────────┘    └──────────────┘    │ (TitaNet)  │    └───────────┘
                                       └────────────┘

    This function uses NeMo's OmegaConf/Hydra config system to build the
    ClusteringDiarizer with swappable model paths.

    Args:
        audio_path: Path to input audio file
        output_dir: Directory for output files
        vad_model: VAD model name or .nemo path
        embedding_model: Speaker embedding model name or .nemo path
        num_speakers: Number of speakers (0 = auto-detect)
        max_num_speakers: Maximum speakers for auto-detection
        device: 'cuda' or 'cpu'
    """
    import torch
    from omegaconf import OmegaConf

    # NeMo diarization imports
    # ClusteringDiarizer is the main cobject that runs the entire cascaded pipeline based on the provided config.
    from nemo.collections.asr.models import ClusteringDiarizer

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Auto-Convert Audio to Mono WAV ---
    audio_path = preprocess_audio(audio_path)

    # --- Create manifest file ---
    manifest_path = os.path.join(output_dir, "input_manifest.json")
    create_manifest(audio_path, manifest_path, num_speakers=num_speakers)

    # --- Build diarization config ---
    # This is the Hydra/OmegaConf config that controls the entire pipeline.
    # Each section maps to a model swap point i.e you can replace the model used in that stage without changing the rest of the pipeline.
    # NeMo configuration is built dynamically

    # Create YAML configuration dictionary for the diarization pipeline. This config will be passed to the ClusteringDiarizer, which will read the settings for each stage of the pipeline (VAD, embeddings, clustering) and execute accordingly.
    config = OmegaConf.create({
        "diarizer": {
            # ── STAGE 1: Voice Activity Detection ──────────────────────
            # SWAP POINT: Change model_path to use a different VAD model
            #   Options: "vad_marblenet", "vad_multilingual_marblenet",
            #            "vad_telephony_marblenet", or path to .nemo file
            # TODO: Understand the hyperparameters with justification (for documentation)
            "vad": {
                "model_path": vad_model,
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.875,
                    "onset": 0.5,
                    "offset": 0.5,
                    "pad_onset": 0.1,
                    "pad_offset": 0.1,
                    "min_duration_on": 0.2,
                    "min_duration_off": 0.2,
                    "filter_speech_first": True,
                },
            },

            # ── STAGE 2: Speaker Embeddings ────────────────────────────
            # SWAP POINT: Change model_path to use different embeddings
            #   Options: "titanet_large", "titanet_small", "ecapa_tdnn",
            #            "speakerverification_speakernet", or path to .nemo file
            "speaker_embeddings": {
                "model_path": embedding_model,
                "parameters": {
                    "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                    "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                    "multiscale_weights": [1, 1, 1, 1, 1],
                    "save_embeddings": False,
                },
            },

            # ── STAGE 3: Clustering ────────────────────────────────────
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": num_speakers > 0,
                    "max_num_speakers": max_num_speakers,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold": 0.15,
                    "sparse_search_volume": 30,
                },
            },

            # ── Input/Output paths ─────────────────────────────────────
            "manifest_filepath": manifest_path,
            "out_dir": output_dir,

            # ── OPTIONAL: MSDD refinement ──────────────────────────────
            # SWAP POINT: Set model_path to use MSDD post-processing
            "msdd_model": {
                "model_path": msdd_model,
                "parameters": {
                    "use_speaker_model_from_ckpt": False,
                    "diar_window_length": 50,
                    "sigmoid_threshold": 0.7,
                    "overlap_infer_spk_limit": 5,
                },
            },

            # ── Oracle settings ────────────────────────────────────────
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
        },

        "sample_rate": 16000,
        "num_workers": 0,
        "verbose": True,
        "device": device,
    })

    # Override num_speakers if specified
    # If num_speakers is set to 0, the clustering will attempt to auto-detect the number of speakers.
    if num_speakers > 0:
        config.diarizer.clustering.parameters.oracle_num_speakers = True

    logger.info("=" * 60)
    logger.info("CASCADED PIPELINE CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  VAD model:       {vad_model}")
    logger.info(f"  Embedding model: {embedding_model}")
    logger.info(f"  MSDD model:      {msdd_model}")
    logger.info(f"  Num speakers:    {num_speakers if num_speakers > 0 else 'auto-detect'}")
    logger.info(f"  Max speakers:    {max_num_speakers}")
    logger.info(f"  Device:          {device}")
    logger.info("=" * 60)

    # --- Run the pipeline ---
    logger.info("🚀 Initializing ClusteringDiarizer...")
    # Measure Time
    start_time = time.time()

    # Create the ClusteringDiarizer object with the provided configuration. 
    diarizer = ClusteringDiarizer(cfg=config)
    logger.info("🎙️  Running diarization...")
    # Run Diarization.
    diarizer.diarize()

    # Runtime Calculation
    elapsed = time.time() - start_time
    logger.info(f"⏱️  Pipeline completed in {elapsed:.2f} seconds")

    # --- Check for output files ---
    # Search for RTTM files (recursively) in the output directory and log their paths.
    rttm_files = list(Path(output_dir).glob("**/*.rttm"))
    if rttm_files:
        for rttm_file in rttm_files:
            logger.info(f"💾 RTTM output: {rttm_file}")
            # Also create CSV from RTTM
            csv_path = str(rttm_file).replace(".rttm", ".csv")
            rttm_to_csv(str(rttm_file), csv_path)
    else:
        logger.warning("⚠️  No RTTM output files found.")

    logger.info("🎉 Cascaded pipeline complete!")


# Post-processing: Convert NeMo's RTTM output to a more user-friendly CSV format.
def rttm_to_csv(rttm_path: str, csv_path: str):
    """Convert RTTM file to CSV format."""
    import pandas as pd

    rows = []
    with open(rttm_path, "r") as f:
        for line in f:
            # Remove leading/trailing whitespace and split the line into parts based on whitespace. This will create a list of elements in the line.
            parts = line.strip().split()

            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                rows.append({
                    "speaker": speaker,
                    "start_time": round(start, 3),
                    "end_time": round(start + duration, 3),
                    "duration": round(duration, 3),
                })

    if rows:
        df = pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"💾 CSV saved: {csv_path}")
        logger.info(f"   {len(df)} segments, {df['speaker'].nunique()} speakers")


def parse_args():
    # Create ArgumentParser object to handle command-line arguments.
    # description appears when user runs python run_cascaded.py --help
    parser = argparse.ArgumentParser(
        description="Run cascaded speaker diarization (VAD → Embeddings → Clustering)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Default pipeline:
  python scripts/run_cascaded.py --audio data/toy-example.wav

  # Swap embedding model to TitaNet-Small:
  python scripts/run_cascaded.py --audio data/my-audio.wav \\
      --embedding-model titanet_small

  # Known number of speakers:
  python scripts/run_cascaded.py --audio data/meeting.wav --num-speakers 3
        """,
    )
    parser.add_argument("--audio", "-a", required=True, help="Path to input audio file")
    parser.add_argument("--vad-model", default="vad_marblenet", help="VAD model name or .nemo path")
    parser.add_argument("--embedding-model", default="titanet_large", help="Embedding model name or .nemo path")
    parser.add_argument("--msdd-model", default=None, help="MSDD model name or .nemo path")
    parser.add_argument("--num-speakers", type=int, default=0, help="Number of speakers (0=auto)")
    parser.add_argument("--max-speakers", type=int, default=8, help="Max speakers for auto-detection")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", default="./outputs/nemo", help="Output directory")
    parser.add_argument("--config", default=None, help="YAML config file path")
    
    # Return Namesspace object containing the parsed command-line arguments.
    return parser.parse_args()


def main():
    args = parse_args()

    # Load YAML config if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        vad_model = config.get("diarizer", {}).get("vad", {}).get("model_path", args.vad_model)
        emb_model = config.get("diarizer", {}).get("speaker_embeddings", {}).get("model_path", args.embedding_model)
        num_speakers = config.get("num_speakers", args.num_speakers)
        max_speakers = config.get("max_num_speakers", args.max_speakers)
        device_setting = config.get("device", args.device)
    else:
        vad_model = args.vad_model
        emb_model = args.embedding_model
        num_speakers = args.num_speakers
        max_speakers = args.max_speakers
        device_setting = args.device

    device = detect_device(device_setting)

    run_cascaded_pipeline(
        audio_path=args.audio,
        output_dir=args.output_dir,
        vad_model=vad_model,
        embedding_model=emb_model,
        msdd_model=args.msdd_model,
        num_speakers=num_speakers,
        max_num_speakers=max_speakers,
        device=device,
    )


if __name__ == "__main__":
    main()
