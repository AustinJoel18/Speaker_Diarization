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
        "vad": "vad_multilingual_marblenet",
        "embedding": "titanet_large",
    },
    {
        "audio_dir": AUDIO_DIR_DENOISED,
        "audio_type": "denoised_audios",
        "vad": "vad_multilingual_marblenet",
        "embedding": "ecapa_tdnn",
    },
]
    

# Function definition for ClusteringDiarizer
def run_experiment(experiment):

    audio_dir = experiment["audio_dir"]
    audio_type = experiment["audio_type"]
    vad = experiment["vad"]
    emb = experiment["embedding"]

    for audio_file in audio_dir.glob("*.wav"):

        print("\n" + "=" * 80)
        print(f"Processing: {audio_file}")

        
        output_dir = (
            f"outputs/diarize_all_new/"
            f"{audio_type}/"
            "auto_num-speakers/"
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

            # if use_fixed_speakers:
            #     cmd.extend([
            #         "--num-speakers",
            #         str(num_speakers),
            #     ])

        print("Running:", " ".join(cmd))

        try:
            subprocess.run(
                cmd,
                check=True,
            )

        except Exception as e:
            print(f"FAILED: {audio_file}")
            print(e)

# # Function definition for NeuralDiarizer
# def run_neural_batch_script(
#     audio_dir,
#     speaker_dict,
#     audio_type,
#     use_fixed_speakers,
# ):
#     """
#     audio_type:
#         "raw_audios"
#         "denoised_audios"

#     use_fixed_speakers:
#         True  -> use --num-speakers
#         False -> auto detect
#     """
#     # Specify the num_speakers mode
#     mode = (
#         "fixed_num-speakers" if use_fixed_speakers
#         else "auto_num-speakers"
#     )

#     for audio_name, num_speakers in speaker_dict.items():

#         audio_file = audio_dir / audio_name

#         if not audio_file.exists():
#             print(f"Audio file not found: {audio_file}")
#             continue

#         print("\n" + "=" * 80)
#         print(f"Processing: {audio_file}")

#         for vad, emb, msdd in itertools.product(
#             VAD_MODELS,
#             EMBED_MODELS,
#             MSDD_MODELS,
#         ):

#             output_dir = (
#                 f"Testing_folder/new_params/NeuralDiarizer_Test6/"
#                 f"{audio_type}/"
#                 f"{mode}/"
#                 f"{vad}/"
#                 f"{emb}"
#             )
            
#             # Create the output directory if it doesn't exist; Not necessary, but good just practice haha
#             Path(output_dir).mkdir(
#                 parents=True,
#                 exist_ok=True,
#             )

#             cmd = [
#                 "python",
#                 "scripts/run_nemo_msdd_diarization.py",
#                 "--audio",
#                 str(audio_file),
#                 "--vad-model",
#                 vad,
#                 "--embedding-model",
#                 emb,
#                 "--msdd-model",
#                 msdd,
#                 "--output-dir",
#                 output_dir,
#                 "--device",
#                 "cuda",
#             ]

#             if use_fixed_speakers:
#                 cmd.extend([
#                     "--num-speakers",
#                     str(num_speakers),
#                 ])

#             print("Running:", " ".join(cmd))

#             try:
#                 subprocess.run(
#                     cmd,
#                     check=True,
#                 )

#             except Exception as e:
#                 print(f"FAILED: {audio_file}")
#                 print(e)

for experiment in EXPERIMENTS:
    run_experiment(experiment)