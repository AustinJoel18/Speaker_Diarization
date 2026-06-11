import itertools # For creating Cartesian product of model combinations
import subprocess # For running the diarization script as a subprocess
from pathlib import Path

#TODO
# AUDIO_DIR = Path("/home/interactionlab/Downloads/Nemo-Cascaded-Diarization/data")
AUDIO_DIR = Path("/media/interactionlab/One Touch/ASD_Dataset/all-audios-denoised")

#TODO
AUDIO_SPEAKERS = {
    "p5-s10_denoised.wav": 4,
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

# For fixed number of speakers
# for audio_name, num_speakers in AUDIO_SPEAKERS.items():

#     audio_file = AUDIO_DIR / audio_name

#     if not audio_file.exists():
#         print(f"Audio file not found: {audio_file}")
#         continue

#     print("\n"+"="*80)
#     print(f"Processing: {audio_file} ({num_speakers} speakers)")

#     for vad, emb, msdd in itertools.product(
#         VAD_MODELS,
#         EMBED_MODELS,
#         MSDD_MODELS
#     ):

#         output_dir = (
#             f"outputs/denoised_audios/fixed_num-speakers/"
#             f"{vad}/"
#             f"{emb}/"
#             f"{msdd or 'no_msdd'}"
#         )

#         cmd = [
#             "python",
#             "scripts/run_nemo_cascaded_diarization.py",
#             "--audio",
#             str(audio_file),
#             "--vad-model",
#             vad,
#             "--embedding-model",
#             emb,
#             "--num-speakers",
#             str(num_speakers),
#             "--output-dir",
#             output_dir,
#             "--device",
#             "cuda",
#         ]

#         if msdd:
#             cmd.extend([
#                 "--msdd-model",
#                 msdd
#             ])

#         print("Running:", " ".join(cmd))

#         try:
#             subprocess.run(cmd, check=True)
#         except Exception as e:
#             print(f"FAILED: {audio_file}")
#             print(e)

# For auto number of speakers
for audio_name, num_speakers in AUDIO_SPEAKERS.items():

    audio_file = AUDIO_DIR / audio_name

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
            f"outputs/testing/auto_num-speakers_new/" #TODO
            f"{vad}/"
            f"{emb}/"
            f"{msdd or 'no_msdd'}"
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