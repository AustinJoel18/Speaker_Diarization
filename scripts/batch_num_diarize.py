from pathlib import Path
import pandas as pd
import subprocess

mapping = pd.read_csv("scripts/speaker_counts.csv")

for _, row in mapping.iterrows():

    audio_file = row["audio_file"]
    num_speakers = int(row["num_speakers"])

    print(f"Processing {audio_file} ({num_speakers} speakers)")

    subprocess.run([
        "python",
        "scripts/run_nemo_cascaded_diarization.py",
        "--audio",
        str(Path(r"/media/interactionlab/One Touch/ASD_Dataset/all-audios-denoised") / audio_file),
        "--num-speakers",
        str(num_speakers)
    ])