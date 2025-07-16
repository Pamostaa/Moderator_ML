from transformers import CamembertForSequenceClassification, CamembertTokenizer
import torch
import pymongo
import time
import gdown
import os
import shutil

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Google Drive folder ID
FOLDER_ID = "11V1m38rQkd5b2Kxme7LoO6IGyRPVd07r"  # Extracted from your link

# Temporary directory to store downloaded model files
TEMP_DIR = "./temp_model"

def download_folder_from_drive(folder_id, output_dir):
    """Download all files from a Google Drive folder."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Use gdown to download the folder
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)
    return output_dir

def load_model_and_tokenizer():
    """Download and load the model and tokenizer from Google Drive."""
    # Download the folder contents
    download_dir = download_folder_from_drive(FOLDER_ID, TEMP_DIR)

    # Assume the folder contains model and tokenizer files directly
    # If the folder has subdirectories, adjust the paths accordingly
    model_dir = download_dir  # Update if model files are in a subdirectory
    tokenizer_dir = download_dir  # Update if tokenizer files are in a subdirectory

    # Load the model and tokenizer
    model = CamembertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = CamembertTokenizer.from_pretrained(tokenizer_dir)
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

severity_map = {'faible': 0, 'moyen': 1, 'élevé': 2}

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

def process_reclamation(description, issue_type=None):
    if not description:
        return None, None
    inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    severity = list(severity_map.keys())[prediction]
    solution = suggest_solution(description, severity, issue_type)
    return severity, solution

if __name__ == '__main__':
    # Clean up temporary directory on script start (optional)
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    client = pymongo.MongoClient("mongodb+srv://ranimabbessi:Y3whR02H5skyatm2@cluster0.mtilp.mongodb.net/legy?retryWrites=true&w=majority")
    db = client["legy"]
    collection = db["orderIssues"]

    print("Starting reclamation processing service...")
    while True:
        # Find pending reclamations (those without a 'severity' field and with status 'PENDING')
        pending = collection.find({
            "severity": {"$exists": False},
            "status": "PENDING"
        }).limit(10)  # Process in batches of 10
        processed_count = 0
        for doc in pending:
            description = doc.get("description")
            issue_type = doc.get("type")
            severity, solution = process_reclamation(description, issue_type)
            if severity and solution:
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"severity": severity, "solution": solution, "status": "RESOLVED"}}
                )
                processed_count += 1
        if processed_count > 0:
            print(f"Processed {processed_count} reclamations.")
        time.sleep(60)  # Check for new reclamations every 60 seconds