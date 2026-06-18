import itertools # For creating Cartesian product of model combinations
import subprocess # For running the diarization script as a subprocess
from pathlib import Path

#TODO
# AUDIO_DIR = Path("/home/interactionlab/Downloads/Nemo-Cascaded-Diarization/data")
AUDIO_DIR_DENOISED = Path("/media/interactionlab/One Touch/ASD_Dataset/all-audios-denoised")
AUDIO_DIR_RAW = Path("/media/interactionlab/One Touch/ASD_Dataset/all-audios")

#TODO
AUDIO_SPEAKERS_DENOISED = {
    "p5-s8_denoised.wav": 4,
    # "p5-s10_denoised.wav": 4,
    # "p7-s8_denoised.wav": 5,
    "p7-s16_denoised.wav": 4,
    "p9-s3-1_denoised.wav": 6,
    # "p9-s9_denoised.wav": 4,
    # "p11-s4_denoised.wav": 4,
    # "p11-s11_denoised.wav": 3,
    # "p12-s3_denoised.wav": 4,
    # "p12-s6_denoised.wav": 3,
    # "p17-s2_denoised.wav": 5,
    # "p17-s6_denoised.wav": 4,
    # "p18-s15_denoised.wav": 3,
    "p18-s17_denoised.wav": 3,
}

AUDIO_SPEAKERS_RAW = {
    "p5-s8.wav": 4,
    # "p5-s10.wav": 4,
    # "p7-s8.wav": 5,
    "p7-s16.wav": 4,
    "p9-s3-1.wav": 6,
    # "p9-s9.wav": 4,
    # "p11-s4.wav": 4,
    # "p11-s11.wav": 3,
    # "p12-s3.wav": 4,
    # "p12-s6.wav": 3,
    # "p17-s2.wav": 5,
    # "p17-s6.wav": 4,
    # "p18-s15.wav": 3,
    "p18-s17.wav": 3,
}

VAD_MODELS = [
    "vad_marblenet",
    "vad_multilingual_marblenet",
    "vad_telephony_marblenet",
]

EMBED_MODELS = [
    "titanet_large",
    "ecapa_tdnn",
]

MSDD_MODELS = [
    "diar_msdd_telephonic",
]

# Function definition for ClusteringDiarizer
def run_clustering_batch_script(
    audio_dir,
    speaker_dict,
    audio_type,
    use_fixed_speakers,
):
    """
    audio_type:
        "raw_audios"
        "denoised_audios"

    use_fixed_speakers:
        True  -> use --num-speakers
        False -> auto detect
    """
    # Specify the num_speakers mode
    mode = (
        "fixed_num-speakers" if use_fixed_speakers
        else "auto_num-speakers"
    )

    for audio_name, num_speakers in speaker_dict.items():

        audio_file = audio_dir / audio_name

        if not audio_file.exists():
            print(f"Audio file not found: {audio_file}")
            continue

        print("\n" + "=" * 80)
        print(f"Processing: {audio_file}")

        for vad, emb in itertools.product(
            VAD_MODELS,
            EMBED_MODELS,
        ):

            output_dir = (
                f"outputs/ClusteringDiarizer/"
                f"{audio_type}/"
                f"{mode}/"
                f"{vad}/"
                f"{emb}"
            )
            
            # Create the output directory if it doesn't exist; Not necessary, but good just practice haha
            Path(output_dir).mkdir(
                parents=True,
                exist_ok=True,
            )

            cmd = [
                "python",
                "scripts/run_nemo_cascaded_diarization.py",
                "--audio",
                str(audio_file),
                "--vad-model",
                vad,
                "--embedding-model",
                emb,
                "--output-dir",
                output_dir,
                "--device",
                "cuda",
            ]

            if use_fixed_speakers:
                cmd.extend([
                    "--num-speakers",
                    str(num_speakers),
                ])

            print("Running:", " ".join(cmd))

            try:
                subprocess.run(
                    cmd,
                    check=True,
                )

            except Exception as e:
                print(f"FAILED: {audio_file}")
                print(e)

# Function definition for NeuralDiarizer
def run_neural_batch_script(
    audio_dir,
    speaker_dict,
    audio_type,
    use_fixed_speakers,
):
    """
    audio_type:
        "raw_audios"
        "denoised_audios"

    use_fixed_speakers:
        True  -> use --num-speakers
        False -> auto detect
    """
    # Specify the num_speakers mode
    mode = (
        "fixed_num-speakers" if use_fixed_speakers
        else "auto_num-speakers"
    )

    for audio_name, num_speakers in speaker_dict.items():

        audio_file = audio_dir / audio_name

        if not audio_file.exists():
            print(f"Audio file not found: {audio_file}")
            continue

        print("\n" + "=" * 80)
        print(f"Processing: {audio_file}")

        for vad, emb, msdd in itertools.product(
            VAD_MODELS,
            EMBED_MODELS,
            MSDD_MODELS,
        ):

            output_dir = (
                f"outputs/NeuralDiarizer/"
                f"{audio_type}/"
                f"{mode}/"
                f"{vad}/"
                f"{emb}"
            )
            
            # Create the output directory if it doesn't exist; Not necessary, but good just practice haha
            Path(output_dir).mkdir(
                parents=True,
                exist_ok=True,
            )

            cmd = [
                "python",
                "scripts/run_nemo_msdd_diarization.py",
                "--audio",
                str(audio_file),
                "--vad-model",
                vad,
                "--embedding-model",
                emb,
                "--msdd-model",
                msdd,
                "--output-dir",
                output_dir,
                "--device",
                "cuda",
            ]

            if use_fixed_speakers:
                cmd.extend([
                    "--num-speakers",
                    str(num_speakers),
                ])

            print("Running:", " ".join(cmd))

            try:
                subprocess.run(
                    cmd,
                    check=True,
                )

            except Exception as e:
                print(f"FAILED: {audio_file}")
                print(e)

''' CLUSTERING DIARIZER EXPERIMENTS '''

# ==========================================================
# 1. DENOISED_AUDIO + FIXED_NUM-SPEAKERS
# ==========================================================

run_clustering_batch_script(
    audio_dir=AUDIO_DIR_DENOISED,
    speaker_dict=AUDIO_SPEAKERS_DENOISED,
    audio_type="denoised_audios",
    use_fixed_speakers=True,
)

# ==========================================================
# 2. DENOISED_AUDIO + AUTO_NUM-SPEAKERS
# ==========================================================

run_clustering_batch_script(
    audio_dir=AUDIO_DIR_DENOISED,
    speaker_dict=AUDIO_SPEAKERS_DENOISED,
    audio_type="denoised_audios",
    use_fixed_speakers=False,
)

# ==========================================================
# 3. RAW_AUDIOS + FIXED_NUM-SPEAKERS
# ==========================================================

run_clustering_batch_script(
    audio_dir=AUDIO_DIR_RAW,
    speaker_dict=AUDIO_SPEAKERS_RAW,
    audio_type="raw_audios",
    use_fixed_speakers=True,
)

# ==========================================================
# 4. RAW_AUDIOS + AUTO_NUM-SPEAKERS
# ==========================================================

run_clustering_batch_script(
    audio_dir=AUDIO_DIR_RAW,
    speaker_dict=AUDIO_SPEAKERS_RAW,
    audio_type="raw_audios",
    use_fixed_speakers=False,
)

''' NEURAL DIARIZER EXPERIMENTS '''
# ==========================================================
# 1. DENOISED_AUDIO + FIXED_NUM-SPEAKERS
# ==========================================================

run_neural_batch_script(
    audio_dir=AUDIO_DIR_DENOISED,
    speaker_dict=AUDIO_SPEAKERS_DENOISED,
    audio_type="denoised_audios",
    use_fixed_speakers=True,
)

# ==========================================================
# 2. DENOISED_AUDIO + AUTO_NUM-SPEAKERS
# ==========================================================

run_neural_batch_script(
    audio_dir=AUDIO_DIR_DENOISED,
    speaker_dict=AUDIO_SPEAKERS_DENOISED,
    audio_type="denoised_audios",
    use_fixed_speakers=False,
)

# ==========================================================
# 3. RAW_AUDIOS + FIXED_NUM-SPEAKERS
# ==========================================================

run_neural_batch_script(
    audio_dir=AUDIO_DIR_RAW,
    speaker_dict=AUDIO_SPEAKERS_RAW,
    audio_type="raw_audios",
    use_fixed_speakers=True,
)

# ==========================================================
# 4. RAW_AUDIOS + AUTO_NUM-SPEAKERS
# ==========================================================

run_neural_batch_script(
    audio_dir=AUDIO_DIR_RAW,
    speaker_dict=AUDIO_SPEAKERS_RAW,
    audio_type="raw_audios",
    use_fixed_speakers=False,
)