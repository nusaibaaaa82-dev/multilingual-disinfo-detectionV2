import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, TrainerCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import json

MODEL_NAME = "roberta-base"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
OUTPUT_DIR = "./roberta_immigrant"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  Device: {device}")
use_bf16 = (device == "mps")

print("\n📥 Loading dataset...")
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
df = pd.DataFrame(dataset['train'])
print(f"   Full: {len(df):,} rows")

immigrant_cols = ['target_origin_immigrant', 'target_origin_undocumented', 'target_origin_migrant_worker']
available = [c for c in immigrant_cols if c in df.columns]
mask = pd.Series(False, index=df.index)
for col in available:
    mask = mask | (df[col] > 0)

df_imm = df[mask].copy()
df_imm['label'] = (df_imm['hate_speech_score'] > 0.5).astype(int)
df_imm = df_imm[df_imm['text'].notna() & (df_imm['text'].str.len() > 10)]
print(f"   Immigrant: {len(df_imm):,} rows")
print(f"   Labels: {df_imm['label'].value_counts().to_dict()}")

train_df, temp_df = train_test_split(df_imm, test_size=0.2, random_state=42, stratify=df_imm['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
print(f"\n✂️  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

print("\n🔤 Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=MAX_LEN)

train_ds = Dataset.from_pandas(train_df[['text', 'label']]).map(tokenize, batched=True)
val_ds = Dataset.from_pandas(val_df[['text', 'label']]).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df[['text', 'label']]).map(tokenize, batched=True)

for ds in [train_ds, val_ds, test_ds]:
    ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

print("\n🤖 Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
    id2label={0: "not-hate", 1: "hate"},
    label2id={"not-hate": 0, "hate": 1},
)
print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")

labels_train = np.array(train_df['label'])
class_weights = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f"   Class weights: {class_weights.tolist()}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none",
    bf16=use_bf16,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    dataloader_num_workers=0,
)

class SaveCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, **kw):
        if model is None: return
        path = os.path.join(OUTPUT_DIR, f"epoch_{int(state.epoch)}")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print(f"💾 Saved → {path}")

trainer = WeightedTrainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    class_weights=class_weights,
    callbacks=[SaveCallback()],
)

print("\n✅ Trainer ready")
print("\n🚀 Training started...")
trainer.train()

best_path = os.path.join(OUTPUT_DIR, "best_roberta_immigrant")
trainer.save_model(best_path)
tokenizer.save_pretrained(best_path)
print(f"\n✅ Best model saved → {best_path}")

print("\n📊 Evaluating...")
out = trainer.predict(test_ds)
yp = np.argmax(out.predictions, axis=-1)
yt = out.label_ids

macro_f1 = f1_score(yt, yp, average="macro", zero_division=0)
weighted = f1_score(yt, yp, average="weighted", zero_division=0)
acc = accuracy_score(yt, yp)
gap = acc - macro_f1

print("\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)
print(f"Accuracy:    {acc:.4f}")
print(f"Macro-F1:    {macro_f1:.4f}")
print(f"Weighted-F1: {weighted:.4f}")
print(f"Gap:         {gap:.4f}")
print()
print(classification_report(yt, yp, digits=4, zero_division=0, target_names=["not-hate", "hate"]))

results = {
    "model": MODEL_NAME, "device": device,
    "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
    "test_accuracy": float(acc), "test_macro_f1": float(macro_f1),
    "test_weighted_f1": float(weighted), "gap": float(gap),
}
with open(os.path.join(OUTPUT_DIR, "results_immigrant.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved")
