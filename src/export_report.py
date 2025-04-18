from fpdf import FPDF
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# 📥 Daten laden
df = pd.read_csv('data/combined_emails.csv')

# 🧹 Leere oder fehlerhafte Werte entfernen
df = df.dropna(subset=['clean_text', 'label'])
df = df[df['clean_text'].str.strip() != '']

# 🔁 Modell und Vektorizer laden
vectorizer = joblib.load('models/vectorizer.pkl')
model = joblib.load('models/model.pkl')

# 🔠 Features und Labels vorbereiten
X = vectorizer.transform(df['clean_text'])
y = df['label']

# 🧪 Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📈 Nur Testdaten vorhersagen
y_pred = model.predict(X_test)

# 📊 Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 🕒 Zeitstempel
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 🔐 Unicode-sicherer PDF-Text
def safe_text(text):
    return ''.join(char if ord(char) < 128 else '?' for char in text)

# 📄 PDF Report erstellen
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.cell(200, 10, txt=safe_text("Phishing Detector - Modellreport"), ln=True, align='C')
pdf.ln(10)
pdf.cell(200, 10, txt=f"Erstellt am: {now}", ln=True)
pdf.ln(10)
pdf.cell(200, 10, txt=f"Accuracy: {accuracy:.4f}", ln=True)

pdf.ln(10)
pdf.cell(200, 10, txt="Classification Report:", ln=True)
for line in report.split('\n'):
    pdf.cell(200, 8, txt=safe_text(line.strip()), ln=True)

pdf.ln(10)
pdf.cell(200, 10, txt="Confusion Matrix:", ln=True)
for row in conf_matrix:
    pdf.cell(200, 8, txt=safe_text(str(row)), ln=True)

# 💾 Speichern im reports-Ordner
pdf.output("reports/model_report.pdf")

print("✅ PDF wurde erfolgreich gespeichert unter: reports/model_report.pdf")
