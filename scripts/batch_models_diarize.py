import itertools
import subprocess
from pathlib import Path

AUDIO_DIR = Path("/media/interactionlab/One Touch/ASD_Dataset/all-audios-denoised")

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
    None,
    "diar_msdd_large",
    "diar_msdd_telephonic",
]

for audio_file in AUDIO_DIR.glob("*.wav"):

    for vad, emb, msdd in itertools.product(
        VAD_MODELS,
        EMBED_MODELS,
        MSDD_MODELS
    ):

        output_dir = (
            f"outputs/"
            f"{vad}/"
            f"{emb}/"
            f"{msdd or 'no_msdd'}"
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