# Note: to run this on lambda machine, make sure to follow the instructions here on installing ffmpeg: https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import pandas as pd
import argparse
from dotenv import load_dotenv
from pathlib import Path
import os
from tqdm import tqdm
from datetime import datetime
import torch
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN,
)

 # Change this to "cuda" if on lambda or more recent GPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

pipeline.to(torch.device(DEVICE))


def diarize(file_or_path: str, num_speakers: int):
    # run the pipeline on an audio file or a bunch of audio files
    if file_or_path.endswith(".wav") and not file_or_path.startswith("._"):
        with ProgressHook() as hook:
            if num_speakers == 0:
                diarization = pipeline(file_or_path,hook=hook,)
            else:
                diarization = pipeline(file_or_path,hook=hook,num_speakers=num_speakers,)
        
        audio_name = Path(file_or_path).stem

        df = {
            "speaker": [],
            "speech_start": [],
            "speech_end": []
        }

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            df["speaker"].append(speaker)
            df["speech_start"].append(turn.start)
            df["speech_end"].append(turn.end)

        csv_path = CSV_DIR / f"{audio_name}.csv"

        pd.DataFrame(df).to_csv(csv_path, index=False)

        rttm_path = RTTM_DIR / f"{audio_name}.rttm"

        with open(rttm_path, "w") as f:
            diarization.write_rttm(f)

        print(f"Saved {csv_path}")
        print(f"Saved {rttm_path}")

    else:
        for file in tqdm(os.listdir(file_or_path)):
            if file.endswith(".wav") and not file.startswith("._"):

                audio_path = os.path.join(file_or_path, file)
                with ProgressHook() as hook:

                    if num_speakers == 0:
                        diarization = pipeline(audio_path,hook=hook,)
                    else:
                        diarization = pipeline(audio_path,hook=hook,num_speakers=num_speakers,)
                
                audio_name = Path(file).stem

                df = {
                    "speaker": [],
                    "speech_start": [],
                    "speech_end": []
                }

                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    df["speaker"].append(speaker)
                    df["speech_start"].append(turn.start)
                    df["speech_end"].append(turn.end)

                csv_path = CSV_DIR / f"{audio_name}.csv"

                pd.DataFrame(df).to_csv(csv_path, index=False)

                rttm_path = RTTM_DIR / f"{audio_name}.rttm"

                with open(rttm_path, "w") as f:
                    diarization.write_rttm(f)

                print(f"✓ Saved {audio_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', dest='file_or_path', type=str, help='Add path to folder of audio files or path to a single audio file')
    parser.add_argument('--ns', dest='num_speakers', type=int, help='Add the integer number of speakers in this audio file. Only applicable for individual audio files atm.')
    parser.add_argument("--output-dir", default="outputs/PyAnnote", type=str, help="Directory to save CSV and RTTM outputs",
                        )
    args = parser.parse_args()

    #Output directories for diarization results
    OUTPUT_DIR = Path(args.output_dir)

    CSV_DIR = OUTPUT_DIR / "csv"
    RTTM_DIR = OUTPUT_DIR / "rttm"

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    RTTM_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    print(f"STARTING AT: {start_time}")
    diarize(args.file_or_path, args.num_speakers)
    end_time = datetime.now()

    print(f"run_pyannote_dirization.py TOOK: {end_time - start_time} SECONDS")