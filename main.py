import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from preprocessor import load_and_prepare_data, preprocess  # استيراد مشترك ✅

print(" Naive Bayes (Multinomial NB) Implementation")

# =========================
# Load & Preprocess (Shared)
# =========================
df = load_and_prepare_data("spam.csv")
print(f" Dataset loaded: {len(df)} samples")

# =========================
# Split data
# =========================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], 
    test_size=0.2, random_state=42, stratify=df['label']
)

print(f" Train: {len(X_train)}, Test: {len(X_test)}")

# =========================
# TF-IDF Vectorization (Traditional ML)
# =========================
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(" TF-IDF Vectorization completed")

# =========================
# Naive Bayes Model
# =========================
model = MultinomialNB(alpha=0.1)  # Tuned parameter
model.fit(X_train_vec, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(" NAIVE BAYES RESULTS")
print("="*50)
print(f" Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# =========================
# Save model & vectorizer
# =========================
import joblib
joblib.dump(model, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'naive_bayes_vectorizer.pkl')
print("\n Model & Vectorizer saved!")

# =========================
# Interactive Demo
# =========================
print("\n" + "="*50)
print("INTERACTIVE SPAM DETECTOR")
print("="*50)

test_messages = [
    "Hey, how are you doing today?",
    "Congratulations! You've won $1000! Click here now!",
    "Meeting at 3pm tomorrow",
    "URGENT: Your account will be suspended! Verify now!",
    "Pizza delivery in 30 minutes"
]

print(" Testing sample messages:")
for msg in test_messages:
    processed = preprocess(msg)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    print(f" '{msg[:50]}...' → {'Spam ' if pred == 1 else 'Ham '}")

print("\n Ready for custom testing (type 'exit' to stop):")
while True:
    text = input("\nEnter message: ").strip()
    if text.lower() in ['exit', 'quit', 'q']:
        break
    
    processed = preprocess(text)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    print(f" Input: {text}")
    print(f" Prediction: {'Spam ' if pred == 1 else 'Ham '}")
    print(f" Confidence: Spam {prob[1]:.2%} | Ham {prob[0]:.2%}")
    print("-" * 50)