
import pandas as pd
import torch
import numpy as np
from transformers import CamembertTokenizer, CamembertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
df = pd.read_csv("order_issues_dummy_french_20000.csv", encoding="utf-8")

# Encode severity
severity_map = {'faible': 0, 'moyen': 1, 'élevé': 2}
df['severity_label'] = df['severity'].map(severity_map)

# Class weights
class_weights = torch.tensor([1.0, 1.2, 1.0], dtype=torch.float).to(device)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['description'], df['severity_label'], test_size=0.2, random_state=42
)

# Load CamemBERT
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForSequenceClassification.from_pretrained(
    "camembert-base", num_labels=3, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1
).to(device)

# Tokenize data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# Create dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels.values)
test_dataset = Dataset(test_encodings, test_labels.values)

# Custom Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
    def get_eval_dataloader(self, eval_dataset=None):
        return torch.utils.data.DataLoader(
            eval_dataset or self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

# Training arguments
training_args = TrainingArguments(
    output_dir="C:/Users/21655/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Moderator_ML/results",
    logging_dir="C:/Users/21655/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Moderator_ML/logs",
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.1,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    overwrite_output_dir=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {'f1': f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average='weighted')}
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print("Evaluation results:", results)

# Classification report
predictions = trainer.predict(test_dataset)
print("\nClassification Report:\n", classification_report(test_labels, np.argmax(predictions.predictions, axis=1), target_names=['faible', 'moyen', 'élevé']))

# Save model
model.save_pretrained("C:/Users/21655/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Moderator_ML/severity_camembert_improved")
tokenizer.save_pretrained("C:/Users/21655/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Moderator_ML/severity_camembert_improved")
