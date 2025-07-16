import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import re

# Download French stopwords
nltk.download('stopwords')
french_stopwords = set(stopwords.words('french'))

# Load dataset
df = pd.read_csv("order_issues_dummy_french_corrected.csv", encoding="utf-8")

# Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in french_stopwords)
    return text

df['description_clean'] = df['description'].apply(clean_text)

# Encode severity
severity_map = {'faible': 0, 'moyen': 1, 'élevé': 2}
df['severity_label'] = df['severity'].map(severity_map)

# Features and target
X_text = df['description_clean']
y = df['severity_label']

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(X_text).toarray()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("F1-score (weighted):", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['faible', 'moyen', 'élevé']))

# Predict on new description
new_description = "La pizza était gâtée, totalement inacceptable !"
new_description_clean = clean_text(new_description)
new_X = tfidf.transform([new_description_clean]).toarray()
prediction = model.predict(new_X)[0]
print(f"\nPredicted severity for '{new_description}': {list(severity_map.keys())[prediction]}")