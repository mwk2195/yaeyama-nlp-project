import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, EarlyStoppingCallback

df = pd.read_csv('Data/hatoma_training_data.csv')

dataset = Dataset.from_pandas(df[['noisy_input', 'clean_katakana_target']])

dataset = dataset.train_test_split(test_size=0.1, seed=42)

model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess_func(examples):
    inputs = examples["noisy_input"]
    targets = examples["clean_katakana_target"]

    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_func, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./byt5-hatoma-normalizer",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    num_train_epochs=15, 
    predict_with_generate=True,
    fp16=False,
    bf16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    label_smoothing_factor=0.1,
    warmup_ratio=0.1,
    save_total_limit=3,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Training started")
trainer.train()

trainer.save_model("./final_hatoma_normalizer")
print("Model saved")