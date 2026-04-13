import os
import json
import urllib.request
import tempfile
import requests
import soundfile as sf
import traceback
from datasets import load_dataset, Audio

# --- Configuration ---
DATASET_NAME = "atlasia/DODa-audio-dataset"
API_URL = "http://localhost:8000/transcribe"
OUTPUT_FILE = "Evaluation/inference_results.json"
# Set to None if you want to evaluate the whole dataset
MAX_SAMPLES = 20 

def main():
    print(f"Loading dataset from Hugging Face: {DATASET_NAME} ...")
    try:
        # The DODa audio dataset might be large, streaming=False will download it.
        # Ensure you have logged in using `huggingface-cli login` if it's restricted.
        dataset = load_dataset(DATASET_NAME, split="train")
        
        # Bypass decoding to avoid torchcodec/ffmpeg crashes
        dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        traceback.print_exc()
        return

    results = []
    
    # Ensure Evaluation folder exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Load previously processed data to resume if script was stopped
    if os.path.exists(OUTPUT_FILE):
        print(f"Loading existing progress from {OUTPUT_FILE}")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                print("JSON decode error on existing file. Starting fresh.")
                results = []
    
    processed_ids = {str(r.get("id")) for r in results if r.get("id") is not None}

    print(f"Successfully loaded dataset with {len(dataset)} total samples.")
    print("Generating transcriptions via API...\n")

    for i, sample in enumerate(dataset):
        if MAX_SAMPLES and i >= MAX_SAMPLES:
            print(f"-- Reached limit of {MAX_SAMPLES} samples --")
            break
            
        sample_id = str(sample.get("id", i))
        
        # Skip if we already processed this
        if sample_id in processed_ids:
            continue
            
        # Extract Arabic script transcription (using the exact DODa column name)
        reference_text = sample.get("darija_Arab_new", "")
        
        # Verify audio data exists
        audio_data = sample.get("audio")
        if not audio_data:
            print(f"[Warning] No audio data for sample {sample_id}, skipping.")
            continue

        # Write raw bytes to a temporary file (bypassing the need for soundfile/ffmpeg)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tmp_path = tmp_audio.name
            if "bytes" in audio_data and audio_data["bytes"]:
                tmp_audio.write(audio_data["bytes"])
            elif "path" in audio_data and audio_data["path"]:
                with open(audio_data["path"], "rb") as f_in:
                    tmp_audio.write(f_in.read())
            
        try:
            
            # Send file to the transcription endpoint
            with open(tmp_path, "rb") as f:
                files = {"audio": (os.path.basename(tmp_path), f, "audio/wav")}
                
                # Make POST request
                response = requests.post(API_URL, files=files)
                
            if response.status_code == 200:
                predicted_text = response.json().get("transcription", "")
                
                # Append structured result
                record = {
                    "id": sample_id,
                    "reference_text": reference_text,
                    "predicted_text": predicted_text
                }
                results.append(record)
                
                # Progressively save
                with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
                    json.dump(results, out_f, ensure_ascii=False, indent=2)
                    
                print(f"[OK] Sample {sample_id} | Result: {predicted_text}")
            else:
                print(f"[ERROR] API failed on sample {sample_id} with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"[EXCEPTION] Failed to process sample {sample_id}: {e}")
        finally:
            # Cleanup the temporary wave file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print(f"\nDone! Inference results saved progressively to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
