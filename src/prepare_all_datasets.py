import pandas as pd
import sys
import os

# Preprocessing-Funktion importieren
sys.path.append(os.path.abspath('./src'))
from preprocessing import clean_text

# 📨 1. Originaldaten laden (emails.csv)
df_emails = pd.read_csv('data/emails.csv')
df_emails = df_emails[['body', 'label']].dropna(subset=['body'])
df_emails['clean_text'] = df_emails['body'].apply(clean_text)
print(f"📦 emails.csv: {len(df_emails)} E-Mails")

# 📁 2. dataset1.csv laden
df1_raw = pd.read_csv('data/dataset1.csv', low_memory=False)
if 'Body' in df1_raw.columns and 'Label' in df1_raw.columns:
    df1 = df1_raw[['Body', 'Label']].rename(columns={"Body": "body", "Label": "label"})
elif 'body' in df1_raw.columns and 'label' in df1_raw.columns:
    df1 = df1_raw[['body', 'label']]
else:
    raise ValueError("❌ dataset1.csv enthält keine erkennbaren Spalten 'body' und 'label'.")
df1 = df1.dropna(subset=['body'])
df1['clean_text'] = df1['body'].apply(clean_text)
print(f"📦 dataset1.csv: {len(df1)} E-Mails")

# 🎯 3. Nur Phishing-Mails aus dataset2.csv extrahieren
df2_raw = pd.read_csv('data/dataset2.csv')
df3 = df2_raw[df2_raw['Email Type'] == 'Phishing Email'].copy()
df3 = df3.rename(columns={"Email Text": "body"})
df3 = df3.dropna(subset=['body'])
df3['label'] = 1
df3['clean_text'] = df3['body'].apply(clean_text)
print(f"📦 phishing from dataset2.csv: {len(df3)} E-Mails")

# 🧩 4. Alles kombinieren
df_combined = pd.concat([df_emails, df1, df3], ignore_index=True)
print(f"🔗 Kombiniert (vor Duplikat-Entfernung): {len(df_combined)}")

# 🔁 Doppelte Zeilen basierend auf clean_text entfernen
df_combined = df_combined.drop_duplicates(subset='clean_text')

# 🧼 Entferne Zeilen mit leerem oder fehlendem Text
df_combined = df_combined.dropna(subset=['clean_text'])
df_combined = df_combined[df_combined['clean_text'].str.strip() != '']
print(f"🧹 Nach Entfernen von Duplikaten und Leereinträgen: {len(df_combined)}")

# 💾 Speichern
df_combined.to_csv('data/combined_emails.csv', index=False, encoding='utf-8')
print(f"✅ Alle Datensätze wurden bereinigt, vereinheitlicht und gespeichert in data/combined_emails.csv")

