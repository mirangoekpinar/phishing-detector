import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib
from preprocessing import clean_text  # aus src importieren

# Daten laden
df = pd.read_csv('../data/emails.csv')

# Text bereinigen
df['clean_text'] = df['body'].apply(clean_text)

# Features und Labels
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell trainieren
model = MultinomialNB()
model.fit(X_train, y_train)

# Modell und Vektorizer speichern
joblib.dump(model, '../models/model.pkl')
joblib.dump(vectorizer, '../models/vectorizer.pkl')
