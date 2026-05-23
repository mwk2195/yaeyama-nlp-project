import pandas as pd
import jiwer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def evaluate_subset(name, df):
    """Calculates and prints Exact Match and CER for a dataset."""
    targets = df['gold_standard'].tolist()
    predictions = df['model_output'].tolist()
    
    exact_matches = sum(1 for t, p in zip(targets, predictions) if t == p)
    exact_match_rate = (exact_matches / len(targets)) * 100
    cer = jiwer.cer(" ".join(targets), " ".join(predictions))
    
    print(f"--- {name.upper()} DATASET ({len(targets)} words) ---")
    print(f"Exact Match Rate: {exact_match_rate:.2f}%")
    print(f"Character Error Rate (CER): {cer:.4f}\n")

def main():
    print("Loading PyTorch model and tokenizer...")
    
    # Load the ByT5 Normalizer and its associated tokenizer
    model_path = "./final_hatoma_normalizer"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Push to GPU if available for faster inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model successfully loaded on: {device.upper()}\n")

    def normalize_text(text):
        """Helper function to run a string through the ByT5 model."""
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=64)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Load the Datasets
    try:
        df_wild = pd.read_csv('evaluation_data/wild.csv')
        df_synthetic = pd.read_csv('evaluation_data/synthetic.csv')
        print("CSV files loaded successfully. Beginning inference...\n")
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return

    # Generate Predictions
    print("Normalizing Wild Text...")
    df_wild['model_output'] = df_wild['messy'].apply(normalize_text)
    
    print("Normalizing Synthetic Text...")
    df_synthetic['model_output'] = df_synthetic['messy'].apply(normalize_text)
    print("Inference complete.\n")

    # Calculate and Print Stratified Metrics
    evaluate_subset("Wild Text", df_wild)
    evaluate_subset("Synthetic OOV Text", df_synthetic)

    # Calculate Total Metrics
    df_total = pd.concat([df_wild, df_synthetic], ignore_index=True)
    evaluate_subset("Total Combined Pipeline", df_total)

    # Save the outputs
    df_total.to_csv('evaluation_data/final_predictions_log.csv', index=False, encoding='utf-8')
    print("Saved exact model outputs to 'evaluation_data/final_predictions_log.csv' for error analysis.")

if __name__ == "__main__":
    main()