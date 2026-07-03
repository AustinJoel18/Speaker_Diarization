import subprocess # For running the diarization script as a subprocess
from pathlib import Path

#TODO
# AUDIO_DIR = Path("/home/interactionlab/Downloads/Nemo-Cascaded-Diarization/data")
AUDIO_DIR_DENOISED = Path("/media/interactionlab/One Touch/ASD_Dataset/all-audios-denoised")
AUDIO_DIR_RAW = Path("/media/interactionlab/One Touch/ASD_Dataset/all-audios")


EXPERIMENTS = [
    {
        "audio_dir": AUDIO_DIR_RAW,
        "audio_type": "raw_audios",
    },
]
    

# Function definition for ClusteringDiarizer
def run_pyannote_experiment(experiment):

    audio_dir = experiment["audio_dir"]
    audio_type = experiment["audio_type"]

    for audio_file in audio_dir.glob("*.wav"):

        print("\n" + "=" * 80)
        print(f"Processing: {audio_file}")

        
        output_dir = (
            f"pyannote_outputs/MP4"
        )
            
        # Create the output directory if it doesn't exist; Not necessary, but good just practice haha
        Path(output_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

        cmd = [
            "python",
            "scripts/run_pyannote_diarization.py",
            "--p",
            str(audio_file),
            "--output-dir",
            output_dir,
            "--ns",
            "0",
        ]

        print("Running:", " ".join(cmd))

        try:
            subprocess.run(
                cmd,
                check=True,
            )

        except Exception as e:
            print(f"FAILED: {audio_file}")
            print(e)



for experiment in EXPERIMENTS:
    run_pyannote_experiment(experiment)