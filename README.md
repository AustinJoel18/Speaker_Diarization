# NeMo Speaker Diarization

Welcome to the **Speaker Diarization** project! This codebase takes an audio file with multiple people talking (like a podcast, debate, or interview), and automatically maps out **who spoke when**.

This uses the industry-standard **NVIDIA NeMo Cascaded Pipeline**, which involves:
1. **Voice Activity Detection (VAD):** Finding exactly where speech happens (ignoring silence).
2. **Speaker Embeddings:** Creating a unique voiceprint for each small slice of audio.
3. **Clustering:** Grouping those voiceprints together to label them (e.g., `speaker_0`, `speaker_1`).

---

## 🚀 1. Initial Setup (Run this only once)

Before you can run the code, you need to install the required Python environment. We have completely automated this for you.

Open a terminal (or Command Prompt) in this exact directory and run the setup script for your OS:

**Mac / Linux:**
```bash
bash bash/setup_env.sh
source ./miniconda/bin/activate nemo
python scripts/patch_nemo.py
```

**Windows (Standard Command Prompt):**
```bat
bash\setup_env.bat
call conda activate nemo
python scripts\patch_nemo.py
```

You're done with setup!

---

## 🎙️ 2. How to Diarize Your Own Audio

1. Take any audio file (e.g. `.mp3`, `.mp4`, `.wav`, etc.) and place it inside the `data/` folder.
2. Open your terminal, make sure your environment is activated (`source ./miniconda/bin/activate nemo` on Mac/Linux, or `call conda activate nemo` on Windows).
3. Run the AI on your audio file:

**Mac / Linux:**
```bash
python scripts/run_cascaded.py --audio data/your-audio-file.mp3
```

**Windows:**
```bat
python scripts\run_cascaded.py --audio data\your-audio-file.mp3
```

*(Note: If you give it an MP3 or Stereo file, the script will automatically convert it into a Mono WAV file for you behind the scenes!)*

### Where are my results?
The pipeline will run and automatically save the results inside the `outputs/nemo/pred_rttms/` folder. 
You will see two files for your audio:
- **`your-audio-file.csv`**: A clean, readable spreadsheet with columns for `speaker, start_time, end_time, duration`.
- **`your-audio-file.rttm`**: A raw text file formatted in the standard format used by speech researchers.

---

## 📁 3. Understanding the Project Structure

This project has been cleaned up to be as simple as possible. Here is what everything does:

*   **`bash/`**: Contains the setup script (`setup_env.sh`) that builds the environment.
*   **`data/`**: Put all your raw `.wav` audio files in here!
*   **`outputs/`**: After you run the pipeline, all your generated results (`.csv` and `.rttm`) will magically appear here.
*   **`scripts/`**: 
    *   `run_cascaded.py`: The main engine! You run this script to process your audio.
    *   `patch_nemo.py`: A helper script that fixes known bugs in NVIDIA's source code so it can run flawlessly on your machine.
*   **`.env.example`**: Example file for setting up advanced environment variables (you can mostly ignore this).
*   **`requirements.txt`**: A list of Python libraries needed by the setup script.

---

## ⚙️ 4. Advanced: Changing Variables

If you want to tweak how the AI works, open `scripts/run_cascaded.py` in a text editor and look at the top of the file around **Line 26**. You can change these variables:

*   **`num_speakers`**: By default, this is set to `0` (auto-detect). If you KNOW there are exactly 3 people in your audio, change this to `3` to force the AI to find exactly 3 voices.
*   **`max_num_speakers`**: The absolute maximum number of people the AI is allowed to guess (default is `8`).
*   **`vad_model`**: The AI model used for Voice Activity Detection. (Default: `"vad_marblenet"`).
*   **`embedding_model`**: The AI model used for extracting voiceprints. (Default: `"titanet_large"`). If you have a local `.nemo` model file, you can change this variable to the absolute path of your local file!
*   **`device`**: Set to `"cpu"` or `"cuda"`. On Mac, it will automatically use CPU.

Enjoy experimenting with Speaker Diarization!
