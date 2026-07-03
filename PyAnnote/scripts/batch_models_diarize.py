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
    "p5-s10_denoised.wav": 4,
    "p7-s8_denoised.wav": 5,
    "p7-s16_denoised.wav": 4,
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

AUDIO_SPEAKERS_RAW = {
    "p5-s8.wav": 4,
    "p5-s10.wav": 4,
    "p7-s8.wav": 5,
    "p7-s16.wav": 4,
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


def run_pyannote_batch_script(
    audio_dir,
    speaker_dict,
    audio_type,
    use_fixed_speakers,
):

    mode = (
        "fixed_num-speakers"
        if use_fixed_speakers
        else "auto_num-speakers"
    )

    for audio_name, num_speakers in speaker_dict.items():

        audio_file = audio_dir / audio_name

        if not audio_file.exists():
            print(f"Missing: {audio_file}")
            continue

        print("=" * 80)
        print(audio_file)

        output_dir = (
            f"pyannote_outputs/"
            f"{audio_type}/"
            f"{mode}"
        )

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
        ]

        if use_fixed_speakers:
            cmd.extend([
                "--ns",
                str(num_speakers),
            ])
        else:
            cmd.extend([
                "--ns",
                "0",
            ])

        print("Running:", " ".join(cmd))

        try:
            subprocess.run(
                cmd,
                check=True,
            )

        except Exception as e:
            print(e)

''' PYANNOTE DIARIZER EXPERIMENTS '''

# ==========================================================
# DENOISED + FIXED
# ==========================================================

run_pyannote_batch_script(
    audio_dir=AUDIO_DIR_DENOISED,
    speaker_dict=AUDIO_SPEAKERS_DENOISED,
    audio_type="denoised_audios",
    use_fixed_speakers=True,
)

# ==========================================================
# DENOISED + AUTO
# ==========================================================

run_pyannote_batch_script(
    audio_dir=AUDIO_DIR_DENOISED,
    speaker_dict=AUDIO_SPEAKERS_DENOISED,
    audio_type="denoised_audios",
    use_fixed_speakers=False,
)

# ==========================================================
# RAW + FIXED
# ==========================================================

run_pyannote_batch_script(
    audio_dir=AUDIO_DIR_RAW,
    speaker_dict=AUDIO_SPEAKERS_RAW,
    audio_type="raw_audios",
    use_fixed_speakers=True,
)

# ==========================================================
# RAW + AUTO
# ==========================================================

run_pyannote_batch_script(
    audio_dir=AUDIO_DIR_RAW,
    speaker_dict=AUDIO_SPEAKERS_RAW,
    audio_type="raw_audios",
    use_fixed_speakers=False,
)
