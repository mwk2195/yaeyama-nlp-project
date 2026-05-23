import pandas as pd
from tokenizers import ByteLevelBPETokenizer

# Load clean corpus
df = pd.read_csv('Data/hatoma_seed_corpus.csv')

# Extract the clean Katakana column into a plain text file
text_file = 'Temp/clean_hatoma_text.txt'
with open(text_file, 'w', encoding='utf-8') as f:
    for text in df['Katakana'].dropna():
        f.write(text + '\n')

# Initialize an empty BPE Tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer
# We set a small vocab size (e.g., 4000) because the corpus is small.
tokenizer.train(
    files=[text_file],
    vocab_size=4000, 
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>"
    ]
)

# Save the tokenizer
tokenizer.save_model(".", "Tokenizer/hatoma_custom_bpe")

print("Tokenizer saved as 'hatoma_custom_bpe-vocab.json' and 'hatoma_custom_bpe-merges.txt'")