import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 📥 Daten laden
df = pd.read_csv('data/combined_emails.csv')

# ❌ Leere entfernen
df = df.dropna(subset=['clean_text', 'label'])
df = df[df['clean_text'].str.strip() != '']

# 🔄 Daten vorbereiten
vectorizer = joblib.load('models/vectorizer.pkl')
X = vectorizer.transform(df['clean_text'])
y = df['label']

# 📊 Aufteilen in Train/Test
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔍 Modell laden
model = joblib.load('models/model.pkl')

# 📈 Vorhersagen und Evaluation
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
