
import pandas as pd
import random
import uuid
from datetime import datetime, timedelta
import numpy as np

# Define categories
issue_types = [
    "LivraisonRetard", "ArticleManquant", "CommandeErronée", "NourritureEndommagée",
    "MauvaiseQualité", "ProblèmePaiement", "ServiceClient", "HygieneProblème",
    "EmballageDéfectueux", "CommandeIncomplète", "RetardRemboursement", "AllergèneNonSignale"
]
severities = ["faible", "moyen", "élevé"]
statuses = ["Ouvert", "Résolu", "Escaladé"]

# Expanded description templates
description_templates = {
    "LivraisonRetard": [
        "La commande a été livrée avec {delay} de retard, nourriture froide et décevante.",
        "Retard de {delay}, service totalement inacceptable !",
        "J'ai attendu {delay}, extrêmement frustrant et incommode.",
        "Livraison en retard de {delay}, qualité du service médiocre.",
        "Commande arrivée après {delay}, c'est scandaleux et perturbant !",
        "Retard de {delay}, satisfaction client compromise.",
        "Petit retard de {delay}, mais la nourriture restait acceptable.",
        "Livraison retardée de {delay}, léger désagrément mais tolérable.",
        "Commande arrivée {delay} en retard, expérience mitigée.",
        "Attente de {delay}, service à améliorer mais pas catastrophique."
    ],
    "ArticleManquant": [
        "Il manque {item} dans ma commande, très décevant et frustrant.",
        "Absence de {item}, remboursement requis immédiatement !",
        "Commande incomplète, {item} non inclus, service à revoir.",
        "Manque de {item}, résolution urgente nécessaire.",
        "Pas reçu {item}, c'est inacceptable et agaçant !",
        "{item} manquant, expérience client médiocre.",
        "Petite omission, {item} n'était pas dans la commande.",
        "Erreur mineure, absence de {item} mais repas correct.",
        "{item} oublié, léger problème mais acceptable.",
        "Veuillez vérifier pourquoi {item} n'a pas été livré."
    ],
    "CommandeErronée": [
        "J'ai reçu {item} au lieu de {correct_item}, erreur frustrante.",
        "Commande incorrecte, livré {item} et non {correct_item}.",
        "J'avais commandé {correct_item}, mais reçu {item}, inacceptable !",
        "Erreur sur la commande : {item} à la place de {correct_item}.",
        "Mauvais article livré, {item} au lieu de {correct_item}, agaçant.",
        "Commande erronée, reçu {item} au lieu de {correct_item}.",
        "Petite confusion, j'ai eu {item} au lieu de {correct_item}.",
        "Erreur légère, livré {item} mais j'avais demandé {correct_item}.",
        "Commande partiellement incorrecte, {item} au lieu de {correct_item}.",
        "{item} livré par erreur, mais le goût était correct."
    ],
    "NourritureEndommagée": [
        "La nourriture était {problem}, totalement immangeable.",
        "{item} est arrivé {problem}, besoin d'un remplacement urgent.",
        "Commande livrée {problem}, qualité inacceptable.",
        "{item} était {problem}, expérience désastreuse.",
        "Tout était {problem}, situation scandaleuse et dangereuse !",
        "Nourriture arrivée {problem}, service déplorable.",
        "{item} légèrement {problem}, mais encore comestible.",
        "Commande un peu {problem}, qualité à améliorer.",
        "{item} était {problem}, expérience mitigée.",
        "Nourriture {problem}, mais pas un gros problème."
    ],
    "MauvaiseQualité": [
        "{item} était {problem}, qualité bien en dessous des attentes.",
        "Mauvais goût de {item}, expérience très décevante.",
        "Qualité horrible de {item}, totalement immangeable.",
        "{item} était {problem}, je ne commanderai plus jamais !",
        "{item} avait un goût {problem}, absolument inacceptable !",
        "Qualité médiocre de {item}, satisfaction compromise.",
        "{item} un peu {problem}, mais acceptable.",
        "Qualité moyenne de {item}, pourrait être mieux.",
        "{item} n'était pas à la hauteur, goût {problem}.",
        "Légère déception avec {item}, qualité {problem}."
    ],
    "ProblèmePaiement": [
        "Facturé deux fois pour la commande, c'est intolérable !",
        "Erreur de paiement, remboursement immédiat requis.",
        "Surfacturation pour {item}, service client inacceptable.",
        "Problème de facturation, très frustrant et gênant.",
        "Petite erreur de paiement, mais à corriger rapidement.",
        "Facturation incorrecte pour {item}, vérification nécessaire.",
        "Léger souci de facturation, mais résolu rapidement.",
        "Erreur mineure sur le paiement, pas un gros problème.",
        "Facture erronée pour {item}, expérience désagréable.",
        "Problème de paiement mineur, mais service à améliorer."
    ],
    "ServiceClient": [
        "Service client injoignable, expérience catastrophique !",
        "Support client très lent, réponse inadéquate et frustrante.",
        "Mauvaise gestion du service client, problème non résolu.",
        "Réponse du service client médiocre, très décevant.",
        "Service client un peu lent, mais réponse acceptable.",
        "Support client correct, mais pourrait être plus efficace.",
        "Légère attente avec le service client, rien de grave.",
        "Service client moyen, réponse utile mais tardive.",
        "Problème mineur avec le support, mais résolu.",
        "Service client a répondu, mais avec du retard."
    ],
    "HygieneProblème": [
        "Présence de {problem} dans {item}, totalement inacceptable !",
        "{item} contenait {problem}, danger sanitaire grave.",
        "Problème d'hygiène avec {item}, investigation urgente requise.",
        "Nourriture contaminée par {problem}, c'est scandaleux !",
        "Léger problème d'hygiène avec {item}, mais comestible.",
        "{item} semblait {problem}, qualité douteuse.",
        "Petite anomalie d'hygiène dans {item}, à vérifier.",
        "Hygiène de {item} légèrement compromise, mais acceptable.",
        "Problème mineur d'hygiène avec {item}, expérience mitigée.",
        "{item} avait un problème d'hygiène, service à améliorer."
    ],
    "EmballageDéfectueux": [
        "Emballage de {item} était {problem}, contenu endommagé.",
        "{item} livré avec un emballage {problem}, inacceptable.",
        "Emballage défectueux, {item} écrasé et immangeable.",
        "Problème d'emballage, {item} était {problem}, frustrant.",
        "Emballage de {item} légèrement {problem}, mais correct.",
        "Petit souci avec l'emballage de {item}, pas grave.",
        "Emballage de {item} un peu {problem}, qualité moyenne.",
        "Léger défaut d'emballage pour {item}, à améliorer.",
        "{item} livré dans un emballage {problem}, expérience mitigée.",
        "Emballage de {item} mal conçu, mais contenu intact."
    ],
    "CommandeIncomplète": [
        "Commande incomplète, manque {item}, très décevant.",
        "Pas reçu {item}, remboursement ou livraison requis.",
        "{item} absent de la commande, service inacceptable.",
        "Manque de {item}, résolution immédiate nécessaire.",
        "Commande partielle, {item} non livré, frustrant.",
        "Oubli de {item}, expérience client médiocre.",
        "Petite omission, {item} manquait mais acceptable.",
        "{item} non inclus, léger désagrément.",
        "Commande sans {item}, mais reste correct.",
        "Erreur mineure, {item} oublié dans la commande."
    ],
    "RetardRemboursement": [
        "Remboursement en retard, service client inacceptable !",
        "Attente de remboursement pour {item}, très frustrant.",
        "Remboursement non traité, c'est scandaleux !",
        "Problème avec le remboursement de {item}, inacceptable.",
        "Retard mineur dans le remboursement, mais gênant.",
        "Remboursement de {item} en attente, à vérifier.",
        "Léger retard de remboursement, mais acceptable.",
        "Remboursement pour {item} un peu lent, pas grave.",
        "Délai de remboursement pour {item} prolongé, à améliorer.",
        "Remboursement en retard, expérience mitigée."
    ],
    "AllergèneNonSignale": [
        "Allergène non signalé dans {item}, danger pour la santé !",
        "Absence d'étiquetage d'allergène pour {item}, inacceptable !",
        "Risque grave, allergène non indiqué dans {item}.",
        "{item} contenait un allergène non signalé, scandaleux !",
        "Allergène non mentionné dans {item}, très dangereux.",
        "Problème d'étiquetage d'allergène pour {item}, frustrant.",
        "Légère erreur d'étiquetage pour {item}, mais risqué.",
        "Allergène non signalé dans {item}, à vérifier.",
        "Problème mineur d'allergène avec {item}, mais préoccupant.",
        "Etiquetage incomplet pour {item}, expérience mitigée."
    ]
}

items = [
    "pizza", "burger", "salade", "frites", "sushi", "pâtes", "sandwich", "dessert",
    "boisson", "sauce", "soupe", "tacos", "wrap", "nuggets", "glace", "riz", "pain"
]
problems = [
    "trempé", "écrasé", "gâté", "froid", "sans goût", "brûlé", "moisi", "mal emballé",
    "contaminé", "périmé", "déchiré", "renversé", "fade", "sous-cuit", "surcuit"
]
delays = ["15 minutes", "30 minutes", "45 minutes", "1 heure", "90 minutes", "2 heures", "3 heures"]

# Expanded keywords for severity
high_severity_keywords = [
    "inacceptable", "scandaleux", "immangeable", "gâté", "plus jamais", "dangereux",
    "horrible", "catastrophique", "intolérable", "dégoûtant", "urgent", "immédiatement",
    "désastreux", "insupportable", "moisi", "allergène", "risque", "santé", "injoignable",
    "périmé", "contaminé", "grave"
]
medium_severity_keywords = [
    "décevant", "frustrant", "mécontent", "90 minutes", "1 heure", "agaçant", "irritant",
    "mauvais service", "pas satisfait", "dérangeant", "problème", "retard important",
    "gênant", "médiocre", "incommode", "insatisfaisant", "erreur notable", "lent",
    "inadéquat", "défectueux"
]
low_severity_keywords = [
    "froid", "manquant", "léger", "petit problème", "15 minutes", "30 minutes", "correct",
    "acceptable", "mineur", "moyen", "pas terrible", "bof", "tolérable", "léger désagrément",
    "un peu", "légèrement", "atténué", "modéré"
]

# Generate data
n_records = 20000
data = {
    "id": [str(uuid.uuid4()) for _ in range(n_records)],
    "orderId": [f"COM-{random.randint(10000, 99999)}" for _ in range(n_records)],
    "userId": [f"USR-{random.randint(1000, 9999)}" for _ in range(n_records)],
    "type": [random.choice(issue_types) for _ in range(n_records)],
    "description": [],
    "reportedAt": [],
    "attachments": [],
    "severity": [],
    "status": [random.choice(statuses) for _ in range(n_records)],
    "resolvedAt": [None] * n_records,
    "resolvedBy": [None] * n_records,
    "issueRef": [f"REF-{random.randint(100000, 999999)}" for _ in range(n_records)]
}

# Generate timestamps (within last year)
start_date = datetime.now() - timedelta(days=365)
for _ in range(n_records):
    days_ago = random.randint(0, 365)
    data["reportedAt"].append((start_date + timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S"))

# Track user complaint frequency
user_complaints = {}
for user in data["userId"]:
    user_complaints[user] = user_complaints.get(user, 0) + 1

# Generate descriptions, severities, and attachments
for i in range(n_records):
    issue_type = data["type"][i]
    template = random.choice(description_templates[issue_type])
    if issue_type == "LivraisonRetard":
        delay = random.choice(delays)
        description = template.format(delay=delay)
    elif issue_type in ["ArticleManquant", "NourritureEndommagée", "MauvaiseQualité", "HygieneProblème", "EmballageDéfectueux", "CommandeIncomplète", "AllergèneNonSignale"]:
        description = template.format(item=random.choice(items), problem=random.choice(problems))
    elif issue_type == "CommandeErronée":
        item, correct_item = random.sample(items, 2)  # Ensure different items
        description = template.format(item=item, correct_item=correct_item)
    else:  # ProblèmePaiement, ServiceClient, RetardRemboursement
        description = template.format(item=random.choice(items)) if "{item}" in template else template
    
    data["description"].append(description)
    
    # Assign severity
    has_attachments = random.choices([0, 1, 2], weights=[0.5, 0.35, 0.15])[0]
    user_complaint_count = user_complaints[data["userId"][i]]
    
    severity_score = 0
    if issue_type in ["NourritureEndommagée", "MauvaiseQualité", "ProblèmePaiement", "HygieneProblème", "AllergèneNonSignale"] and any(word in description.lower() for word in high_severity_keywords):
        severity_score += 3
    elif issue_type in ["ArticleManquant", "CommandeErronée", "CommandeIncomplète"] and any(word in description.lower() for word in ["immédiatement", "inacceptable", "agaçant"]):
        severity_score += 2
    elif issue_type == "LivraisonRetard" and any(word in description.lower() for word in ["2 heures", "3 heures", "scandaleux"]):
        severity_score += 3
    elif issue_type in ["ServiceClient", "RetardRemboursement"] and any(word in description.lower() for word in high_severity_keywords):
        severity_score += 3
    elif any(word in description.lower() for word in medium_severity_keywords):
        severity_score += 2
    elif any(word in description.lower() for word in low_severity_keywords):
        severity_score += 1
    if has_attachments > 0:
        severity_score += 1
    if user_complaint_count > 4:
        severity_score += 1
    
    if severity_score >= 4:
        severity = "élevé"
    elif severity_score == 2 or severity_score == 3:
        severity = "moyen"
    else:
        severity = "faible"
    
    # Adjust for distribution (~50% faible, 30% moyen, 20% élevé)
    if random.random() < 0.5 and severity_score <= 1:
        severity = "faible"
    elif random.random() < 0.3 and severity_score in [2, 3]:
        severity = "moyen"
    
    data["severity"].append(severity)
    
    # Add attachments
    attachments = [f"image_{random.randint(1000, 9999)}.jpg" for _ in range(has_attachments)]
    data["attachments"].append(",".join(attachments))

# Create DataFrame
df = pd.DataFrame(data)

# Add resolvedAt and resolvedBy for Résolu issues
for i, status in enumerate(df["status"]):
    if status == "Résolu":
        reported_at = datetime.strptime(df["reportedAt"][i], "%Y-%m-%d %H:%M:%S")
        df.at[i, "resolvedAt"] = (reported_at + timedelta(hours=random.randint(1, 48))).strftime("%Y-%m-%d %H:%M:%S")
        df.at[i, "resolvedBy"] = f"MOD-{random.randint(1, 50)}"

# Save to CSV
df.to_csv("order_issues_dummy_french_20000.csv", index=False, encoding="utf-8-sig")
print("Generated 20,000 samples and saved to order_issues_dummy_french_20000.csv")
