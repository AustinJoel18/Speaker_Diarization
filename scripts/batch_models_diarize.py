import itertools # For creating Cartesian product of model combinations
import subprocess # For running the diarization script as a subprocess
from pathlib import Path

#TODO
# AUDIO_DIR = Path("/home/interactionlab/Downloads/Nemo-Cascaded-Diarization/data")
AUDIO_DIR1 = Path("/media/interactionlab/One Touch/ASD_Dataset/all-audios-denoised")
AUDIO_DIR2 = Path("/media/interactionlab/One Touch/ASD_Dataset/all-audios")

#TODO
AUDIO_SPEAKERS_1 = {
    "p5-s10_denoised.wav": 4,
    "p7-s8_denoised.wav": 5,
    "p9-s3-1_denoised.wav": 6,
    "p9-s9_denoised.wav": 4,
    "p11-s4_denoised.wav": 4,
    "p11-s11_denoised.wav": 3,
    "p12-s3_denoised.wav": 4,
    "p12-s6_denoised.wav": 3,
    "p17-s2_denoised.wav": 5,
    "p17-s6_denoised.wav": 4,
    "p18-s15_denoised.wav": 3,
    "p18-s17_denoised.wav": 3,
}

AUDIO_SPEAKERS_2= {
    "p5-s10.wav": 4,
    "p7-s8.wav": 5,
    "p9-s3-1.wav": 6,
    "p9-s9.wav": 4,
    "p11-s4.wav": 4,
    "p11-s11.wav": 3,
    "p12-s3.wav": 4,
    "p12-s6.wav": 3,
    "p17-s2.wav": 5,
    "p17-s6.wav": 4,
    "p18-s15.wav": 3,
    "p18-s17.wav": 3,
}

#TODO
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

# For denoised audio and fixed number of speakers
for audio_name, num_speakers in AUDIO_SPEAKERS_1.items():

    audio_file = AUDIO_DIR1 / audio_name

    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        continue

    print("\n"+"="*80)
    print(f"Processing: {audio_file} ({num_speakers} speakers)")

    for vad, emb, msdd in itertools.product(
        VAD_MODELS,
        EMBED_MODELS,
        MSDD_MODELS
    ):

        output_dir = (
            f"outputs/NeuralDiarizer/denoised_audios/fixed_num-speakers/"
            f"{vad}/"
            f"{emb}"
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
            "--num-speakers",
            str(num_speakers),
            "--output-dir",
            output_dir,
            "--device",
            "cuda",
        ]

        if msdd:
            cmd.extend([
                "--msdd-model",
                msdd
            ])

        print("Running:", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"FAILED: {audio_file}")
            print(e)

# For denoised audio and auto number of speakers
for audio_name, num_speakers in AUDIO_SPEAKERS_1.items():

    audio_file = AUDIO_DIR1 / audio_name

    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        continue

    print("\n"+"="*80)
    print(f"Processing: {audio_file} ({num_speakers} speakers)")

    for vad, emb, msdd in itertools.product(
        VAD_MODELS,
        EMBED_MODELS,
        MSDD_MODELS
    ):

        output_dir = (
            f"outputs/NeuralDiarizer/denoised_audios/auto_num-speakers/"
            f"{vad}/"
            f"{emb}"
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
            "--output-dir",
            output_dir,
            "--device",
            "cuda",
        ]

        if msdd:
            cmd.extend([
                "--msdd-model",
                msdd
            ])

        print("Running:", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"FAILED: {audio_file}")
            print(e)

# For raw audio and fixed number of speakers
for audio_name, num_speakers in AUDIO_SPEAKERS_2.items():

    audio_file = AUDIO_DIR2 / audio_name

    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        continue

    print("\n"+"="*80)
    print(f"Processing: {audio_file} ({num_speakers} speakers)")

    for vad, emb, msdd in itertools.product(
        VAD_MODELS,
        EMBED_MODELS,
        MSDD_MODELS
    ):

        output_dir = (
            f"outputs/NeuralDiarizer/raw_audios/fixed_num-speakers/" #TODO
            f"{vad}/"
            f"{emb}/"
        )

        cmd = [
            "python",
            "scripts/run_nemo_msdd_diarization.py", #TODO
            "--audio",
            str(audio_file),
            "--vad-model",
            vad,
            "--embedding-model",
            emb,
            "--num-speakers",
            str(num_speakers),
            "--output-dir",
            output_dir,
            "--device",
            "cuda",
        ]

        if msdd:
            cmd.extend([
                "--msdd-model",
                msdd
            ])

        print("Running:", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"FAILED: {audio_file}")
            print(e)  

# For raw audio and auto number of speakers
for audio_name, num_speakers in AUDIO_SPEAKERS_2.items():

    audio_file = AUDIO_DIR2 / audio_name

    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        continue

    print("\n"+"="*80)
    print(f"Processing: {audio_file} ({num_speakers} speakers)")

    for vad, emb, msdd in itertools.product(
        VAD_MODELS,
        EMBED_MODELS,
        MSDD_MODELS
    ):

        output_dir = (
            f"outputs/NeuralDiarizer/raw_audios/auto_num-speakers/" #TODO
            f"{vad}/"
            f"{emb}/"
        )

        cmd = [
            "python",
            "scripts/run_nemo_msdd_diarization.py", #TODO
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

        if msdd:
            cmd.extend([
                "--msdd-model",
                msdd
            ])

        print("Running:", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"FAILED: {audio_file}")
            print(e)