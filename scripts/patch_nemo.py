import os
import sys

def patch_nemo():
    # Attempt to find the site-packages path for the current environment i.e where NeMo is installed
    try:
        import nemo.collections.asr as asr
        nemo_path = os.path.dirname(asr.__file__)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

    print(f"🔍 Found NeMo installation at: {nemo_path}")

    # Patch 1: audio_to_label.py (Fix PyTorch 2.x 0-D tensor iteration bug)
    # Conflict between PyTorch DataLoader batch sample format and Nemo's audio_to_label.py expected format causes error.
    audio_to_label_path = os.path.join(nemo_path, "data", "audio_to_label.py")
    if os.path.exists(audio_to_label_path):
        with open(audio_to_label_path, "r") as f:
            content = f.read()

        target = "    _, audio_lengths, _, tokens_lengths = zip(*batch)"
        replacement = "    if type(batch) is tuple:\n        batch = [batch]\n    _, audio_lengths, _, tokens_lengths = zip(*batch)"

        if replacement in content:
            print("✅ audio_to_label.py is already patched.")
        elif target in content:
            content = content.replace(target, replacement)
            with open(audio_to_label_path, "w") as f:
                f.write(content)
            print("✅ Patched audio_to_label.py (PyTorch dataloader tuple fix applied)")
        else:
            print("⚠️ Warning: Could not find target code in audio_to_label.py. Patch may be incompatible with this NeMo version.")
    else:
        print("⚠️ Warning: audio_to_label.py not found.")

    # Patch 2: vad_utils.py (Fix macOS fork multiprocessing deadlock)
    # Run DataLoader with num_workers=0 to avoid multiprocessing deadlock due to fork() behavior (which creates multiple subprocesses). Multiprocessing is feasible for Linux.
    vad_utils_path = os.path.join(nemo_path, "parts", "utils", "vad_utils.py")
    if os.path.exists(vad_utils_path):
        with open(vad_utils_path, "r") as f:
            content = f.read()

        # NeMo 1.21.0 sets num_workers = 20 by default which breaks macOS
        target_train = "num_workers=20,"
        target_infer = "num_workers=15,"
        
        changed = False
        if target_train in content:
            content = content.replace(target_train, "num_workers=0,")
            changed = True
        if target_infer in content:
            content = content.replace(target_infer, "num_workers=0,")
            changed = True
            
        if changed:
            with open(vad_utils_path, "w") as f:
                f.write(content)
            print("✅ Patched vad_utils.py (macOS multiprocessing num_workers=0 fix applied)")
        else:
            print("✅ vad_utils.py is already patched or targets not found.")
    else:
        print("⚠️ Warning: vad_utils.py not found.")

    
    # Patch 3: msdd_models.py (Fix GPU MSDD inference tensor device mismatch)
    # NeMo 1.21.0 + modern PyTorch/CUDA can leave MSDD inference
    # inputs on CPU while the model is on GPU, causing:
    # RuntimeError: Input type (torch.FloatTensor) and weight type
    # (torch.cuda.FloatTensor) should be the same.
    msdd_models_path = os.path.join(nemo_path, "models", "msdd_models.py")

    if os.path.exists(msdd_models_path):
        with open(msdd_models_path, "r") as f:
            content = f.read()

        patch_marker = "signals = signals.to(device)"

        if patch_marker in content:
            print("✅ msdd_models.py already patched.")
        else:
            target = """        else:
            with autocast():
                _preds, scale_weights = self.msdd_model.forward_infer(
                    input_signal=signals, input_signal_length=signal_lengths, emb_vectors=emb_vectors, targets=None
                )"""

            replacement = """        else:
            device = next(self.msdd_model.parameters()).device

            signals = signals.to(device)
            signal_lengths = signal_lengths.to(device)

            if emb_vectors is not None:
                emb_vectors = emb_vectors.to(device)

            with autocast(enabled=False):
                _preds, scale_weights = self.msdd_model.forward_infer(
                    input_signal=signals,
                    input_signal_length=signal_lengths,
                    emb_vectors=emb_vectors,
                    targets=None,
                )"""

            if target in content:
                content = content.replace(target, replacement)

                with open(msdd_models_path, "w") as f:
                    f.write(content)

                print("✅ Patched msdd_models.py (MSDD GPU device fix applied)")
            else:
                print("⚠️ Warning: Could not find MSDD inference block in msdd_models.py")
    else:
        print("⚠️ Warning: msdd_models.py not found.")

    print("🎉 NeMo patching complete!")

if __name__ == "__main__":
    patch_nemo()