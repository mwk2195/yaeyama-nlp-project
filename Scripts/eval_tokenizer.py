import pandas as pd
import tiktoken
from tokenizers import ByteLevelBPETokenizer

def evaluate_tokenizers(texts, custom_tokenizer_path="./Tokenizer"):
    print("Loading tokenizers...")
    
    # Load OpenAI's GPT-4 Tokenizer (cl100k_base)
    gpt4_tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Load Custom Hatoma BPE Tokenizer
    custom_tokenizer = ByteLevelBPETokenizer(
    f"{custom_tokenizer_path}/hatoma_custom_bpe-vocab.json",
    f"{custom_tokenizer_path}/hatoma_custom_bpe-merges.txt"
)

    # Metrics Trackers
    gpt4_total_tokens = 0
    gpt4_corrupted_bytes = 0
    
    custom_total_tokens = 0
    custom_corrupted_bytes = 0
    
    total_words = len(texts)

    print(f"Evaluating {total_words} clean Yaeyama words...\n")

    for text in texts:
        # GPT-4 Tokenization 
        gpt4_encoded = gpt4_tokenizer.encode(text)
        gpt4_total_tokens += len(gpt4_encoded)
        
        # Decode back to check for the Unicode Replacement Character ()
        gpt4_decoded = [gpt4_tokenizer.decode([t]) for t in gpt4_encoded]
        gpt4_corrupted_bytes += sum(1 for chunk in gpt4_decoded if '\ufffd' in chunk)

        # Custom Tokenization
        custom_encoded = custom_tokenizer.encode(text)

        custom_total_tokens += len(custom_encoded.ids)

        # Extract the string tokens directly to check for byte-shredding
        custom_decoded = custom_encoded.tokens
        custom_corrupted_bytes += sum(1 for chunk in custom_decoded if '\ufffd' in chunk)

    # Calculate Final Metrics
    gpt4_fertility = gpt4_total_tokens / total_words
    custom_fertility = custom_total_tokens / total_words
    
    # Calculate how much more efficient the custom tokenizer is
    efficiency_gain = ((gpt4_total_tokens - custom_total_tokens) / gpt4_total_tokens) * 100

    print("="*50)
    print("TOKENIZATION EVALUATION REPORT")
    print("="*50)
    
    print("\n--- GPT-4 (cl100k_base) Baseline ---")
    print(f"Total Tokens: {gpt4_total_tokens}")
    print(f"Average Tokens per Word (Fertility): {gpt4_fertility:.2f}")
    print(f"Corrupted Byte Instances (): {gpt4_corrupted_bytes}")

    print("\n--- Custom Hatoma BPE Tokenizer ---")
    print(f"Total Tokens: {custom_total_tokens}")
    print(f"Average Tokens per Word (Fertility): {custom_fertility:.2f}")
    print(f"Corrupted Byte Instances (): {custom_corrupted_bytes}")

    print("\n--- Conclusion ---")
    print(f"The custom tokenizer is {efficiency_gain:.1f}% more efficient than GPT-4.")
    if custom_corrupted_bytes == 0 and gpt4_corrupted_bytes > 0:
        print("The custom tokenizer successfully eliminated all Unicode byte-shredding.")
    print("="*50)

if __name__ == "__main__":
    # Load the clean text from evaluation datasets
    try:
        df_wild = pd.read_csv('evaluation_data/wild.csv')
        df_synthetic = pd.read_csv('evaluation_data/synthetic.csv')
        
        all_clean_words = df_wild['gold_standard'].tolist() + df_synthetic['gold_standard'].tolist()
        
        evaluate_tokenizers(all_clean_words)
        
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")