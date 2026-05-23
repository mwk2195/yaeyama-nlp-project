import pandas as pd
import random
import re

df = pd.read_csv('Data/hatoma_seed_corpus.csv')

def drop_long_vowel(text):
    """Simulates a user forgetting the vowel extender."""
    return text.replace('ー', '')

def swap_long_vowel(text):
    """Simulates a user typing the vowel out instead of using the extender."""

    # A mapping for the preceding vowel sound
    vowel_map = {'ア': 'ア', 'カ': 'ア', 'サ': 'ア', 'タ': 'ア', 'ナ': 'ア', 'ハ': 'ア', 'マ': 'ア', 'ヤ': 'ア', 'ラ': 'ア', 'ワ': 'ア',
                 'イ': 'イ', 'キ': 'イ', 'シ': 'イ', 'チ': 'イ', 'ニ': 'イ', 'ヒ': 'イ', 'ミ': 'イ', 'リ': 'イ',
                 'ウ': 'ウ', 'ク': 'ウ', 'ス': 'ウ', 'ツ': 'ウ', 'ヌ': 'ウ', 'フ': 'ウ', 'ム': 'ウ', 'ユ': 'ウ', 'ル': 'ウ',
                 'エ': 'エ', 'ケ': 'エ', 'セ': 'エ', 'テ': 'エ', 'ネ': 'エ', 'ヘ': 'エ', 'メ': 'エ', 'レ': 'エ',
                 'オ': 'オ', 'コ': 'オ', 'ソ': 'オ', 'ト': 'オ', 'ノ': 'オ', 'ホ': 'オ', 'モ': 'オ', 'ヨ': 'オ', 'ロ': 'オ'}
    
    chars = list(text)
    for i in range(1, len(chars)):
        if chars[i] == 'ー':
            prev_char = chars[i-1]
            # Strip dakuten/handakuten for the lookup
            base_char = prev_char.translate(str.maketrans('ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ', 'カキクケコサシスセソタチツテトハヒフヘホハヒフヘホ'))
            if base_char in vowel_map:
                chars[i] = vowel_map[base_char]
    return "".join(chars)

def flatten_small_kana(text):
    """Simulates a user typing full-sized kana instead of small kana."""
    small_to_large = str.maketrans('ァィゥェォッャュョ', 'アイウエオツヤユヨ')
    return text.translate(small_to_large)

def drop_dakuten(text):
    """Simulates a user dropping voicing marks."""
    dakuten_drop = str.maketrans('ガギグゲゴザジズゼゾダヂヅデドバビブベボ', 'カキクケコサシスセソタチツテトハヒフヘホ')
    return text.translate(dakuten_drop)

synthetic_data = []

for index, row in df.iterrows():
    clean_katakana = str(row['Katakana'])
    phonemic_target = str(row['Phonemic'])

    synthetic_data.append({
        'noisy_input': clean_katakana,
        'clean_katakana_target': clean_katakana,
        'phonemic_target': phonemic_target
    })

    for _ in range(random.randint(1, 3)):
        noisy_text = clean_katakana

        if 'ー' in noisy_text:
            if random.random() < 0.5:
                noisy_text = drop_long_vowel(noisy_text)
            else:
                noisy_text = swap_long_vowel(noisy_text)
        
        if random.random() < 0.3:
            noisy_text = flatten_small_kana(noisy_text)
        
        if random.random() < 0.2:
            noisy_text = drop_dakuten(noisy_text)
        
        if noisy_text != clean_katakana:
            synthetic_data.append({
                'noisy_input': noisy_text,
                'clean_katakana_target': clean_katakana,
                'phonemic_target': phonemic_target
            })

synthetic_df = pd.DataFrame(synthetic_data)
# Shuffle the dataset so the neural network doesn't memorize the alphabetical order 
synthetic_df = synthetic_df.sample(frac=1).reset_index(drop=True)
synthetic_df.to_csv('Data/hatoma_training_data.csv', index=False, encoding='utf-8-sig')

print(f"Generated {len(synthetic_df)} training pairs from {len(df)} base entries.")