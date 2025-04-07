from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['legit', 'phishing'], yticklabels=['legit', 'phishing'])
    plt.xlabel('Vorhergesagt')
    plt.ylabel('Tatsächlich')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('../reports/confusion_matrix.png')
    plt.show()
