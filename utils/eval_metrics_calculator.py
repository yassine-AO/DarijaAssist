import json
import re
import os
try:
    import jiwer
except ImportError:
    print("Please install jiwer to calculate WER. Run: pip install jiwer")
    exit(1)

# --- Configuration ---
INPUT_FILE = "Evaluation/inference_results.json"
OUTPUT_FILE = "Evaluation/evaluation_results.json"

def normalize_text(text: str) -> str:
    """
    Applies standard NLP normalization to Darija/Arabic text to ensure fair evaluation,
    ignoring common textual variations or punctuations.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
        
    # 1. Lowercase (helpful if any Latin characters or numbers are included)
    text = text.lower()
    
    # 2. Remove standard and Arabic punctuation
    text = re.sub(r'[!@#\$%\^&\*\(\)_\+\-\=\{\}\[\]:;"\'<>\?,\.\/\\،\؛\؟]', ' ', text)
    
    # 3. Arabic specific text normalization
    # Remove Diacritics (Tashkeel)
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    
    # Normalize Alef variations to standard Alef
    text = re.sub(r'[إأآا]', 'ا', text)
    
    # Normalize Yaa variations
    text = re.sub(r'[يى]', 'ي', text)
    
    # Normalize Taa Marbuta to Haa (common in Moroccan Darija transcriptions)
    text = re.sub(r'ة', 'ه', text)
    
    # Normalize Waw with Hamza
    text = re.sub(r'ؤ', 'و', text)
    
    # 4. Normalize spacing
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[-] Error: Could not find '{INPUT_FILE}'.")
        print("Please run 'eval_inference_runner.py' first to collect predictions.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if not data:
        print("[-] Data file is empty.")
        return

    evaluated_data = []
    total_wer = 0.0
    total_cer = 0.0
    valid_samples = 0
    
    print("Normalizing reference and prediction texts...")
    print("Calculating Word Error Rate (WER) and Character Error Rate (CER)...\n")
    
    for record in data:
        ref_raw = record.get("reference_text", "")
        pred_raw = record.get("predicted_text", "")
        
        # Apply normalization to BOTH strings
        ref_norm = normalize_text(ref_raw)
        pred_norm = normalize_text(pred_raw)
        
        # We can only compute WER realistically if the reference is not empty
        if ref_norm:
            try:
                wer_score = jiwer.wer(ref_norm, pred_norm)
                cer_score = jiwer.cer(ref_norm, pred_norm)
                
                evaluated_record = {
                    "id": record["id"],
                    "raw_reference": ref_raw,
                    "raw_prediction": pred_raw,
                    "normalized_reference": ref_norm,
                    "normalized_prediction": pred_norm,
                    "wer": wer_score,
                    "cer": cer_score
                }
                evaluated_data.append(evaluated_record)
                
                total_wer += wer_score
                total_cer += cer_score
                valid_samples += 1
            except Exception as e:
                print(f"[!] Evaluation failed for sample {record['id']}: {e}")
        else:
            print(f"[!] Skipping sample {record['id']} because normalized reference is empty.")
    
    # Display and Save summary
    if valid_samples > 0:
        avg_wer = total_wer / valid_samples
        avg_cer = total_cer / valid_samples
        
        # Structure the final output with a summary object
        final_output = {
            "summary": {
                "total_processed_samples": valid_samples,
                "overall_average_wer": avg_wer,
                "overall_average_cer": avg_cer
            },
            "detailed_results": evaluated_data
        }
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
            
        print("="*40)
        print("==== EVALUATION SUMMARY ====")
        print("="*40)
        print(f"Total Processed Samples: {valid_samples}")
        print(f"Overall Average WER:     {avg_wer:.4f}  ({(avg_wer * 100):.2f}%)")
        print(f"Overall Average CER:     {avg_cer:.4f}  ({(avg_cer * 100):.2f}%)")
        print("="*40)
        print(f"\nDetailed metrics saved to: {OUTPUT_FILE}")
    else:
        print("[-] No valid samples were evaluated. Please check your data.")

if __name__ == "__main__":
    main()
