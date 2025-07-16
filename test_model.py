
import pandas as pd
import torch
from transformers import CamembertForSequenceClassification, CamembertTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load saved model and tokenizer
model = CamembertForSequenceClassification.from_pretrained("C:/Users/21655/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Moderator_ML/severity_camembert_improved").to(device)
tokenizer = CamembertTokenizer.from_pretrained("C:/Users/21655/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Moderator_ML/severity_camembert_improved")
severity_map = {'faible': 0, 'moyen': 1, 'élevé': 2}

# Solution suggestion function
def suggest_solution(description, severity, issue_type=None):
    description = description.lower()
    if severity == "élevé":
        if "allergène" in description:
            return "Remboursement complet, investigation urgente, et suivi médical si nécessaire."
        if "gâté" in description or "moisi" in description or "danger" in description:
            return "Remboursement complet et investigation immédiate."
        elif "injoignable" in description or "désastreuse" in description:
            return "Contacter le client directement et offrir un bon de réduction."
        elif "facturé deux fois" in description or "intolérable" in description:
            return "Rembourser le montant excédentaire et offrir une compensation."
        elif issue_type == "LivraisonRetard" and ("3 heures" in description or "scandaleux" in description):
            return "Remboursement complet et excuses officielles."
        return "Remboursement ou remplacement immédiat."
    elif severity == "moyen":
        if issue_type == "ArticleManquant" or "manque" in description or "manquant" in description:
            return "Livrer l'article manquant ou offrir un remboursement partiel."
        elif issue_type == "CommandeErronée" or "erronée" in description or "au lieu de" in description:
            return "Livrer la commande correcte ou offrir un bon de réduction."
        elif issue_type == "LivraisonRetard" or "retard" in description:
            return "Offrir un bon de réduction pour la prochaine commande."
        return "Contacter le client pour résoudre le problème rapidement."
    else:
        return "Envoyer une excuse par e-mail et un petit bon de réduction."

# Load dataset and create test set
df = pd.read_csv("order_issues_dummy_french_20000.csv", encoding="utf-8")
df['severity_label'] = df['severity'].map(severity_map)
from sklearn.model_selection import train_test_split
_, test_df = train_test_split(df, test_size=0.2, random_state=42)  # 4,000 samples
test_df = test_df.sample(n=1000, random_state=42)  # Subsample 1,000
test_texts = test_df['description'].tolist()
test_labels = test_df['severity_label'].tolist()
test_types = test_df['type'].tolist()

# Tokenize test data
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Create test dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item
    def __len__(self):
        return len(self.labels)

test_dataset = Dataset(test_encodings, test_labels)

# Predict
predictions = []
true_labels = []
solutions = []

for i in range(len(test_dataset)):
    inputs = test_dataset[i]
    inputs = {key: val.unsqueeze(0) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    pred_severity = list(severity_map.keys())[pred]
    solution = suggest_solution(test_texts[i], pred_severity, test_types[i])
    predictions.append(pred)
    true_labels.append(test_labels[i])
    solutions.append(solution)
    if i < 10:  # Print first 10
        print(f"Description: '{test_texts[i]}'")
        print(f"Predicted severity: {pred_severity}, True severity: {list(severity_map.keys())[test_labels[i]]}")
        print(f"Suggested solution: {solution}\n")

# Classification report
print("\nClassification Report for Test Set (1,000 samples):\n")
print(classification_report(true_labels, predictions, target_names=['faible', 'moyen', 'élevé'], zero_division=0))

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['faible', 'moyen', 'élevé'], yticklabels=['faible', 'moyen', 'élevé'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (1,000 Samples)')
plt.savefig("C:/Users/21655/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Moderator_ML/results/confusion_matrix_20000_1000_samples.png")
plt.show()

# Save predictions to CSV
results_df = pd.DataFrame({
    "description": test_texts,
    "true_severity": [list(severity_map.keys())[label] for label in test_labels],
    "predicted_severity": [list(severity_map.keys())[p] for p in predictions],
    "suggested_solution": solutions,
    "issue_type": test_types
})
results_df.to_csv("C:/Users/21655/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Moderator_ML/results/test_predictions_20000_1000_samples.csv", index=False, encoding="utf-8-sig")
print("Predictions saved to C:/Users/21655/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Moderator_ML/results/test_predictions_20000_1000_samples.csv")
